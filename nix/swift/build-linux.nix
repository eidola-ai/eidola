# Minimal Swift toolchain for Linux — extracts the official tarball and exposes
# the binaries.  Only tools that work without extra runtime setup (swift-format,
# swift-frontend, etc.) are expected to work out of the box; full compilation
# requires additional system libraries that are not wired up here.
{ lib
, stdenv
, autoPatchelfHook
, src
, version
, ncurses
, zlib
, libxml2
, curl
}:

stdenv.mkDerivation {
  inherit src version;
  name = "swift";

  nativeBuildInputs = [ autoPatchelfHook ];
  buildInputs = [ ncurses zlib libxml2 curl stdenv.cc.cc.lib ];

  sourceRoot = ".";

  unpackPhase = ''
    tar xf $src --strip-components=1
  '';

  installPhase = ''
    mkdir -p $out
    cp -R usr/* $out/
  '';

  # Only verify swift-format works — full compilation needs more wiring
  doCheck = false;
}
