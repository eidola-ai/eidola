//! What the interface asks of its host beyond the core: the shell's own
//! calls (the system browser, the bundle store), the page (a reload, a file
//! of the bundle's own), and the update check.
//!
//! [`ShellHost`] is the one implementation, over a [`ShellClient`]; the page's
//! two calls are handed in by the entry point (`crate::web`), so everything
//! here stays target-independent.

use std::rc::Rc;

use eidola_app_core::AppCore;
use eidola_app_core::error::AppError;
use eidola_app_core::updater::ReleaseCheckError;
use eidola_app_core::updater::mobile::{
    ActivatedBundle, BundleState, BundleUpdate, PendingBundleUpdate,
};
use futures_util::future::LocalBoxFuture;

use crate::client::{ShellClient, Transport};
use crate::protocol::Hello;
use crate::update;

/// Where the update check looks. A release bundle always asks the real
/// release feed; a development bundle may be handed a fixture
/// (`crate::dev`), and nothing else can.
pub type CheckUpdate =
    Rc<dyn Fn(Rc<AppCore>) -> LocalBoxFuture<'static, Result<BundleUpdate, ReleaseCheckError>>>;

/// The interface's host.
pub trait Host: 'static {
    /// The shell's handshake, as it answered at boot.
    fn hello(&self) -> &Hello;
    /// Hand an `https:` URL to the system browser (`shell.openExternal`).
    fn open_external(&self, url: String) -> LocalBoxFuture<'static, Result<(), String>>;
    /// `bundle.state`.
    fn bundle_state(&self) -> LocalBoxFuture<'static, Result<BundleState, String>>;
    /// Find and verify the next release; stages nothing.
    fn check_update(
        &self,
        core: Rc<AppCore>,
    ) -> LocalBoxFuture<'static, Result<BundleUpdate, ReleaseCheckError>>;
    /// The user approved `pending`: download, check, stage and activate it.
    fn install(
        &self,
        core: Rc<AppCore>,
        pending: PendingBundleUpdate,
    ) -> LocalBoxFuture<'static, Result<ActivatedBundle, AppError>>;
    /// Make the embedded bundle the one the next load boots.
    fn revert(&self) -> LocalBoxFuture<'static, Result<BundleState, AppError>>;
    /// Load the document again (`location.reload()`): the shell then serves
    /// whichever bundle is active.
    fn reload(&self);
    /// A text file of the bundle this document was served from, by its
    /// bundle path (the licences).
    fn bundle_text(&self, path: &'static str) -> LocalBoxFuture<'static, Result<String, String>>;
}

/// The page's own calls, which only the entry point can make.
pub struct Page {
    pub reload: Box<dyn Fn()>,
    pub fetch_text: Box<dyn Fn(&'static str) -> LocalBoxFuture<'static, Result<String, String>>>,
}

/// The host over a shell.
pub struct ShellHost<T> {
    client: ShellClient<T>,
    hello: Hello,
    page: Page,
    check: CheckUpdate,
}

impl<T: Transport> ShellHost<T> {
    pub fn new(client: ShellClient<T>, hello: Hello, page: Page) -> Self {
        Self {
            client,
            hello,
            page,
            check: Rc::new(|core: Rc<AppCore>| Box::pin(async move { update::check(&core).await })),
        }
    }

    /// A development bundle's update source (`crate::dev`).
    #[cfg(feature = "dev")]
    pub fn with_update_check(mut self, check: CheckUpdate) -> Self {
        self.check = check;
        self
    }
}

impl<T: Transport> Host for ShellHost<T> {
    fn hello(&self) -> &Hello {
        &self.hello
    }

    fn open_external(&self, url: String) -> LocalBoxFuture<'static, Result<(), String>> {
        let client = self.client.clone();
        Box::pin(async move { client.open_external(&url).await.map_err(|e| e.to_string()) })
    }

    fn bundle_state(&self) -> LocalBoxFuture<'static, Result<BundleState, String>> {
        let client = self.client.clone();
        Box::pin(async move { client.bundle_state().await.map_err(|e| e.to_string()) })
    }

    fn check_update(
        &self,
        core: Rc<AppCore>,
    ) -> LocalBoxFuture<'static, Result<BundleUpdate, ReleaseCheckError>> {
        (self.check)(core)
    }

    fn install(
        &self,
        core: Rc<AppCore>,
        pending: PendingBundleUpdate,
    ) -> LocalBoxFuture<'static, Result<ActivatedBundle, AppError>> {
        let client = self.client.clone();
        Box::pin(async move { update::install(&core, &client, pending).await })
    }

    fn revert(&self) -> LocalBoxFuture<'static, Result<BundleState, AppError>> {
        let client = self.client.clone();
        Box::pin(async move { update::revert(&client).await })
    }

    fn reload(&self) {
        (self.page.reload)();
    }

    fn bundle_text(&self, path: &'static str) -> LocalBoxFuture<'static, Result<String, String>> {
        (self.page.fetch_text)(path)
    }
}
