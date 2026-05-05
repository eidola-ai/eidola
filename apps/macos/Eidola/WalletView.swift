//
//  WalletView.swift
//  Eidola
//

import EidolaAppCore
import SwiftUI

struct WalletView: View {
  var core: Core
  @State private var recoveryMessage: String?

  var body: some View {
    List {
      if core.credentials.isEmpty && core.spendingCredentials.isEmpty {
        ContentUnavailableView(
          "No Credentials",
          systemImage: "creditcard.trianglebadge.exclamationmark",
          description: Text("Allocate credits from Account in Settings to get started.")
        )
      } else {
        if !core.spendingCredentials.isEmpty {
          Section {
            ForEach(core.spendingCredentials, id: \.nonce) { cred in
              HStack {
                VStack(alignment: .leading, spacing: 2) {
                  Text(cred.nonce.prefix(16) + "...")
                    .font(.system(.body, design: .monospaced))
                  Text("Stuck \u{2014} \(cred.spendAmount) credits charged")
                    .font(.caption)
                    .foregroundStyle(.orange)
                }

                Spacer()

                Text("\(cred.credits) credits")
                  .fontWeight(.medium)
                  .monospacedDigit()
              }
              .padding(.vertical, 2)
            }
          } header: {
            HStack {
              Text("In Flight")
              Spacer()
              Button("Recover All") {
                Task {
                  let recovered = await core.recoverSpendingCredentials()
                  if recovered.isEmpty {
                    recoveryMessage = "No credentials could be recovered."
                  } else {
                    recoveryMessage = "Recovered \(recovered.count) credential(s)."
                  }
                }
              }
              .buttonStyle(.borderless)
              .disabled(core.isLoading)
            }
          }
        }

        if !core.credentials.isEmpty {
          Section("Active Credentials") {
            ForEach(core.credentials, id: \.nonce) { cred in
              HStack {
                VStack(alignment: .leading, spacing: 2) {
                  Text(cred.nonce.prefix(16) + "...")
                    .font(.system(.body, design: .monospaced))
                  Text("Generation \(cred.generation)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }

                Spacer()

                Text("\(cred.credits) credits")
                  .fontWeight(.medium)
                  .monospacedDigit()
              }
              .padding(.vertical, 2)
            }
          }
        }
      }
    }
    .navigationTitle("Wallet")
    .overlay {
      if core.isLoading {
        ProgressView()
          .frame(maxWidth: .infinity, maxHeight: .infinity)
          .background(.ultraThinMaterial)
      }
    }
    .alert("Recovery", isPresented: Binding(
      get: { recoveryMessage != nil },
      set: { if !$0 { recoveryMessage = nil } }
    )) {
      Button("OK") { recoveryMessage = nil }
    } message: {
      if let msg = recoveryMessage { Text(msg) }
    }
    .task {
      await core.fetchCredentials()
    }
  }
}
