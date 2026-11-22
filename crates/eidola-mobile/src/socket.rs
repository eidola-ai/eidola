//! The shell's TCP pipe as a byte stream: what app-core's host-socket
//! transport dials, runs TLS and attestation over, and speaks HTTP through.
//!
//! - **Receiving is pulled.** A `tcp.receive` is in flight only while a
//!   reader is waiting on an empty buffer, and the shell reads from the
//!   network only while a receive is pending (`shell-contract.md` → TCP pipe),
//!   so a reader that stops reading stops the network: at most one reply's
//!   bytes ([`ShellDialer::receive_max`]) wait here, and the kernel's window
//!   holds the rest.
//! - **Sending is buffered**, as a kernel's send buffer is: a write is taken
//!   into a buffer of at most [`SEND_BUFFER`] bytes and acknowledged; one
//!   `tcp.send` at a time carries what has gathered, so small TLS records
//!   coalesce while a send is in flight, and a writer that outruns the shell
//!   waits for the buffer to drain. A failed send fails the next write or
//!   flush. `poll_flush` returns once the shell has handed every byte to its
//!   operating system.
//! - **`poll_shutdown`** flushes, then sends `tcp.shutdown` (FIN); receiving
//!   continues. **Dropping** the stream posts `tcp.close`.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll, ready};

use tinfoil_verifier::pipe::{BoxFuture, Dialer, PipeStream};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

use crate::client::{ShellClient, Transport};
use crate::protocol::ShellError;

/// The default most bytes one `tcp.receive` returns.
pub const RECEIVE_MAX: usize = 256 << 10;

/// The most bytes a stream holds that the shell has not yet taken.
pub const SEND_BUFFER: usize = 256 << 10;

/// Opens TCP connections through the shell.
pub struct ShellDialer<T> {
    client: ShellClient<T>,
    receive_max: usize,
}

impl<T: Transport> ShellDialer<T> {
    pub fn new(client: ShellClient<T>) -> Self {
        Self {
            client,
            receive_max: RECEIVE_MAX,
        }
    }

    /// The most bytes one `tcp.receive` asks for (1 to 1 MiB, the contract's
    /// bound).
    pub fn receive_max(mut self, max: usize) -> Self {
        self.receive_max = max.clamp(1, 1 << 20);
        self
    }

    /// Connect to `host:port` (a DNS name or an IP literal, brackets
    /// allowed).
    pub async fn connect(&self, host: &str, port: u16) -> Result<ShellSocket<T>, ShellError> {
        let host = host.trim_start_matches('[').trim_end_matches(']');
        let id = self.client.next_socket();
        // A connect abandoned midway (its caller timed out) or refused
        // still closes the id; closing an id the shell forgot is no error.
        let mut guard = CloseOnDrop(Some((self.client.clone(), id)));
        self.client.tcp_connect(id, host, port).await?;
        guard.0 = None;
        Ok(ShellSocket {
            client: self.client.clone(),
            id,
            receive_max: self.receive_max,
            received: Vec::new(),
            read_at: 0,
            eof: false,
            receiving: None,
            unsent: Vec::new(),
            sending: None,
            failed: None,
            shutting: None,
            shut: false,
        })
    }
}

impl<T: Transport> Dialer for ShellDialer<T> {
    fn dial<'a>(&'a self, host: &'a str, port: u16) -> BoxFuture<'a, io::Result<PipeStream>> {
        Box::pin(async move {
            let socket = self.connect(host, port).await?;
            Ok(Box::new(socket) as PipeStream)
        })
    }
}

struct CloseOnDrop<T: Transport>(Option<(ShellClient<T>, u64)>);

impl<T: Transport> Drop for CloseOnDrop<T> {
    fn drop(&mut self) {
        if let Some((client, id)) = self.0.take() {
            drop(client.tcp_close(id));
        }
    }
}

type Pending<R> = BoxFuture<'static, Result<R, ShellError>>;

/// One connection through the shell's pipe.
pub struct ShellSocket<T: Transport> {
    client: ShellClient<T>,
    id: u64,
    receive_max: usize,
    /// The last receive's bytes; `received[read_at..]` is unread.
    received: Vec<u8>,
    read_at: usize,
    eof: bool,
    receiving: Option<Pending<Option<Vec<u8>>>>,
    /// Bytes written and acknowledged, not yet handed to a `tcp.send`.
    unsent: Vec<u8>,
    sending: Option<Pending<()>>,
    /// A failed send or shutdown: every later write fails with it.
    failed: Option<(io::ErrorKind, String)>,
    shutting: Option<Pending<()>>,
    shut: bool,
}

impl<T: Transport> ShellSocket<T> {
    /// The socket id the shell knows this connection by.
    pub fn id(&self) -> u64 {
        self.id
    }

    fn fail(&mut self, error: ShellError) -> io::Error {
        let error = io::Error::from(error);
        self.failed = Some((error.kind(), error.to_string()));
        error
    }

    fn failure(&self) -> Option<io::Error> {
        self.failed
            .as_ref()
            .map(|(kind, message)| io::Error::new(*kind, message.clone()))
    }

    /// Drive the send in flight, and start the next with what has gathered.
    /// Ready once nothing is in flight (and nothing new to start).
    fn poll_send(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        loop {
            if let Some(error) = self.failure() {
                return Poll::Ready(Err(error));
            }
            if let Some(sending) = self.sending.as_mut() {
                let result = ready!(sending.as_mut().poll(cx));
                self.sending = None;
                if let Err(error) = result {
                    return Poll::Ready(Err(self.fail(error)));
                }
            }
            if self.unsent.is_empty() {
                return Poll::Ready(Ok(()));
            }
            let send = self.client.tcp_send(self.id, &self.unsent);
            self.unsent.clear();
            self.sending = Some(send);
        }
    }
}

impl<T: Transport> AsyncRead for ShellSocket<T> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = &mut *self;
        loop {
            if this.read_at < this.received.len() {
                let n = buf.remaining().min(this.received.len() - this.read_at);
                buf.put_slice(&this.received[this.read_at..this.read_at + n]);
                this.read_at += n;
                return Poll::Ready(Ok(()));
            }
            if this.eof || buf.remaining() == 0 {
                return Poll::Ready(Ok(()));
            }
            let receiving = this
                .receiving
                .get_or_insert_with(|| this.client.tcp_receive(this.id, this.receive_max));
            let result = ready!(receiving.as_mut().poll(cx));
            this.receiving = None;
            match result {
                Ok(Some(bytes)) => {
                    this.received = bytes;
                    this.read_at = 0;
                }
                Ok(None) => this.eof = true,
                Err(error) => return Poll::Ready(Err(error.into())),
            }
        }
    }
}

impl<T: Transport> AsyncWrite for ShellSocket<T> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        data: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = &mut *self;
        if data.is_empty() {
            return Poll::Ready(Ok(0));
        }
        if this.shut || this.shutting.is_some() {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "the sending side is shut down",
            )));
        }
        // A full buffer waits for the send in flight.
        while this.unsent.len() >= SEND_BUFFER {
            ready!(this.poll_send(cx))?;
        }
        if let Some(error) = this.failure() {
            return Poll::Ready(Err(error));
        }
        let n = data.len().min(SEND_BUFFER - this.unsent.len());
        this.unsent.extend_from_slice(&data[..n]);
        // Start it now if nothing is in flight; a pending send is driven by
        // the next write or flush, and a failed one is kept for them.
        let _ = this.poll_send(cx);
        Poll::Ready(Ok(n))
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.poll_send(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = &mut *self;
        if this.shut {
            return Poll::Ready(Ok(()));
        }
        let shutting = match this.shutting.as_mut() {
            Some(shutting) => shutting,
            None => {
                ready!(this.poll_send(cx))?;
                this.shutting.insert(this.client.tcp_shutdown(this.id))
            }
        };
        let result = ready!(shutting.as_mut().poll(cx));
        this.shutting = None;
        this.shut = true;
        Poll::Ready(result.map_err(|error| this.fail(error)))
    }
}

impl<T: Transport> Drop for ShellSocket<T> {
    fn drop(&mut self) {
        drop(self.client.tcp_close(self.id));
    }
}
