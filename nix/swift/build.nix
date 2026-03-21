{ lib
, stdenv
, xar
, cpio
, makeWrapper
, src
, version
, bintools
, coreutils
, gnugrep
, gnused
, llvmPackages_19
}:

let
  wrapperParams = rec {
    inherit bintools;

    default_cc_wrapper = stdenv.cc;
    coreutils_bin = lib.getBin coreutils;
    gnugrep_bin = gnugrep;
    gnused_bin = gnused;
    suffixSalt = lib.replaceStrings [ "-" "." ] [ "_" "_" ] stdenv.targetPlatform.config;
    use_response_file_by_default = 1;

    swiftOs = "macosx";
    swiftArch = "arm64";

    swiftLibSubdir = "lib/swift/${swiftOs}";
    swiftModuleSubdir = "lib/swift/${swiftOs}";

    swiftStaticLibSubdir = lib.replaceStrings [ "/swift/" ] [ "/swift_static/" ] swiftLibSubdir;
    swiftStaticModuleSubdir = lib.replaceStrings [ "/swift/" ] [ "/swift_static/" ] swiftModuleSubdir;
  };

in
stdenv.mkDerivation (wrapperParams // {
  inherit src version;
  name = "swift";

  buildInputs = [ makeWrapper xar cpio ];

  phases = [ "unpackPhase" "installPhase" "checkPhase" ];

  unpackPhase = ''
    xar -xf $src
    zcat < swift-''${version}-osx-package.pkg/Payload | cpio -i
  '';

  installPhase = ''
    cp -R . $out
    mkdir -p $out/bin

    for progName in swift-symbolgraph-extract swift-autolink-extract; do
      ln -s $out/usr/bin/swift-frontend $out/bin/$progName
    done

    rm -rf $out/usr/bin/clang $out/usr/bin/clang++ $out/usr/bin/clang-17 $out/usr/bin/clangd $out/usr/bin/lld

    ln -s ${llvmPackages_19.clang}/bin/clang $out/usr/bin/clang-17
    ln -s ${llvmPackages_19.clang}/bin/clang $out/usr/bin/clang
    ln -s ${llvmPackages_19.clang}/bin/clang++ $out/usr/bin/clang++
    ln -s ${llvmPackages_19.clang-unwrapped}/bin/clangd $out/usr/bin/clangd
    ln -s ${llvmPackages_19.lld}/bin/lld $out/usr/bin/lld

    for executable in llvm-ar llvm-cov llvm-profdata; do
      rm -rf $out/usr/bin/$executable
      ln -s ${llvmPackages_19.llvm}/bin/$executable $out/usr/bin/$executable
    done

    swift=$out
    swiftDriver="$out/usr/bin/swift-driver"
    
    for progName in swift swiftc; do
      prog=$out/usr/bin/$progName
      export prog progName swift swiftDriver sdk
      rm $out/usr/bin/$progName
      substituteAll '${./build/wrapper.sh}' $out/bin/$progName
      chmod a+x $out/bin/$progName
    done

    mkdir -p $out/nix-support
    substituteAll ${./build/setup-hook.sh} $out/nix-support/setup-hook

    ln -s $out/usr/lib $out/lib
  '';

  doCheck = false;
})
