# The About window (src/about.rs).
#
# English is the source locale: every message must exist here, and the build
# script generates one typed accessor per message from this file. A message
# missing from another locale falls back to English at runtime; a message that
# exists only in another locale is a build error.

about-title = Eidola

about-version-label = Version
about-version-value = v{ $version }

about-purpose-lead = A quiet page for thinking with a machine — private by construction, not by policy.
about-purpose-attestation = Every request runs inside sealed, hardware-attested enclaves, and this app verifies the cryptographic evidence before a word leaves your machine.

about-source-note = Source available on GitHub.

# The accessible name of the repository link.
about-github = View on GitHub
# The visible link text: the name plus the arrow that marks it as leaving the app.
about-github-cta = View on GitHub →
