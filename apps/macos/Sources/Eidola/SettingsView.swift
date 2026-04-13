//
//  SettingsView.swift
//  Eidola
//

import EidolaAppCore
import SwiftUI

struct SettingsView: View {
  var core: Core

  @State private var baseUrlInput = ""
  @State private var accountIdInput = ""
  @State private var accountSecretInput = ""

  var body: some View {
    Form {
      // Server
      Section("Server") {
        LabeledContent("Base URL") {
          HStack {
            TextField("https://...", text: $baseUrlInput)
              .textFieldStyle(.roundedBorder)
              .frame(minWidth: 200)

            Button("Save") {
              guard !baseUrlInput.isEmpty else { return }
              try? core.setBaseUrl(baseUrlInput)
            }
            .disabled(baseUrlInput.isEmpty)
          }
        }
      }

      // Account credentials
      Section("Account Credentials") {
        if core.configState?.hasAccount == true {
          Label("Credentials configured", systemImage: "checkmark.circle.fill")
            .foregroundStyle(.green)
        } else {
          LabeledContent("Account ID") {
            TextField("UUID", text: $accountIdInput)
              .textFieldStyle(.roundedBorder)
              .frame(minWidth: 200)
          }

          LabeledContent("Secret") {
            SecureField("Secret", text: $accountSecretInput)
              .textFieldStyle(.roundedBorder)
              .frame(minWidth: 200)
          }

          Button("Save Credentials") {
            guard !accountIdInput.isEmpty, !accountSecretInput.isEmpty else { return }
            try? core.setAccountCredentials(id: accountIdInput, secret: accountSecretInput)
            accountIdInput = ""
            accountSecretInput = ""
          }
          .disabled(accountIdInput.isEmpty || accountSecretInput.isEmpty)
        }
      }

      // Attestation
      Section("Attestation") {
        if let state = core.configState {
          LabeledContent("Attestation URL") {
            Text(state.attestationUrl ?? "Default (Tinfoil ATC)")
              .foregroundStyle(.secondary)
          }

          LabeledContent("Trusted measurements") {
            if state.trustedMeasurements.isEmpty {
              Text("None (attestation disabled)")
                .foregroundStyle(.secondary)
            } else {
              Text("\(state.trustedMeasurements.count) measurement(s)")
            }
          }

          if !state.trustedMeasurements.isEmpty {
            ForEach(
              Array(state.trustedMeasurements.enumerated()), id: \.offset
            ) { _, m in
              VStack(alignment: .leading, spacing: 2) {
                Text("SNP: \(m.snp.prefix(32))...")
                  .font(.system(.caption, design: .monospaced))
                Text("RTMR1: \(m.tdxRtmr1.prefix(32))...")
                  .font(.system(.caption, design: .monospaced))
                Text("RTMR2: \(m.tdxRtmr2.prefix(32))...")
                  .font(.system(.caption, design: .monospaced))
              }
              .foregroundStyle(.secondary)
            }
          }

          LabeledContent("Hardware Root CA") {
            Text(state.hasHardwareRootCa ? "Set" : "Not set")
              .foregroundStyle(state.hasHardwareRootCa ? .primary : .secondary)
          }

          LabeledContent("Hardware Intermediate CA") {
            Text(state.hasHardwareIntermediateCa ? "Set" : "Not set")
              .foregroundStyle(state.hasHardwareIntermediateCa ? .primary : .secondary)
          }
        }
      }

      // Domain separator
      Section("Protocol") {
        if let state = core.configState {
          LabeledContent("Domain Separator") {
            Text(state.domainSeparator)
              .font(.system(.caption, design: .monospaced))
              .foregroundStyle(.secondary)
              .textSelection(.enabled)
          }
        }
      }

      // Error display
      if let error = core.errorMessage {
        Section {
          Label(error, systemImage: "exclamationmark.triangle")
            .foregroundStyle(.red)
        }
      }
    }
    .formStyle(.grouped)
    .navigationTitle("Settings")
    .onAppear {
      if let state = core.configState {
        baseUrlInput = state.baseUrl ?? ""
      }
    }
  }
}
