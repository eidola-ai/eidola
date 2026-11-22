# R8 keeps the activity and the page's one JavaScript entry point (WebView looks the
# annotation up at run time, so it stays too); the rest, the Kotlin standard library
# included, is shrunk to what the shell calls.
-keep class ai.eidola.shell.MainActivity { <init>(); }
-keepattributes RuntimeVisibleAnnotations
-keepclasseswithmembers class * { @android.webkit.JavascriptInterface <methods>; }
-dontobfuscate
-dontwarn org.jetbrains.annotations.**
