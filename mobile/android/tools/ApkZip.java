import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.TreeMap;
import java.util.stream.Stream;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/**
 * Writes an unsigned, unaligned APK whose bytes depend only on its inputs:
 * entries sorted by name, all STORED, one fixed timestamp, no extra fields.
 *
 * Usage: java ApkZip.java OUT BASE_APK [zip/name=file | zip/prefix/=dir]...
 */
public class ApkZip {
    // 2000-01-01T00:00:00 as a DOS time; ZipEntry converts it in the default zone, so callers pin UTC.
    static final long FIXED = 946684800000L;

    public static void main(String[] a) throws IOException {
        TreeMap<String, byte[]> entries = new TreeMap<>();
        try (ZipFile base = new ZipFile(a[1])) {
            for (ZipEntry e : java.util.Collections.list(base.entries())) {
                if (e.isDirectory()) continue;
                try (InputStream in = base.getInputStream(e)) { entries.put(e.getName(), in.readAllBytes()); }
            }
        }
        for (int i = 2; i < a.length; i++) {
            String[] kv = a[i].split("=", 2);
            Path src = Paths.get(kv[1]);
            if (kv[0].endsWith("/")) {
                try (Stream<Path> walk = Files.walk(src)) {
                    for (Path p : (Iterable<Path>) walk.filter(Files::isRegularFile)::iterator) {
                        String rel = src.relativize(p).toString().replace('\\', '/');
                        if (rel.startsWith(".") || rel.contains("/.")) continue;
                        entries.put(kv[0] + rel, Files.readAllBytes(p));
                    }
                }
            } else {
                entries.put(kv[0], Files.readAllBytes(src));
            }
        }
        // The manifest first, as tools expect; the rest in sorted order.
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        try (ZipOutputStream z = new ZipOutputStream(buf)) {
            byte[] manifest = entries.remove("AndroidManifest.xml");
            put(z, "AndroidManifest.xml", manifest);
            for (var e : entries.entrySet()) put(z, e.getKey(), e.getValue());
        }
        try (FileOutputStream f = new FileOutputStream(a[0])) { buf.writeTo(f); }
    }

    static void put(ZipOutputStream z, String name, byte[] data) throws IOException {
        ZipEntry e = new ZipEntry(name);
        e.setMethod(ZipEntry.STORED);
        e.setTime(FIXED);
        e.setSize(data.length);
        e.setCompressedSize(data.length);
        CRC32 crc = new CRC32();
        crc.update(data);
        e.setCrc(crc.getValue());
        z.putNextEntry(e);
        z.write(data);
        z.closeEntry();
    }
}
