//
//  AccountView.swift
//  Eidola
//

import EidolaAppCore
import SwiftUI

struct AccountView: View {
  var core: Core

  @State private var allocateAmount = ""

  var body: some View {
    Form {
      // Account status
      Section("Account") {
        if let state = core.configState {
          if state.hasAccount {
            Label("Account configured", systemImage: "checkmark.circle.fill")
              .foregroundStyle(.green)

            Button("Show Account Info") {
              Task { await showAccount() }
            }

            Button("Reset Account", role: .destructive) {
              try? core.resetAccount()
            }
          } else {
            Label("No account", systemImage: "xmark.circle")
              .foregroundStyle(.secondary)

            Button("Create Account") {
              Task { await core.createAccount() }
            }
            .disabled(state.baseUrl == nil)
          }
        }
      }

      // Balances
      Section("Balances") {
        if let balances = core.balances {
          LabeledContent("Available") {
            Text("\(balances.available) credits")
              .monospacedDigit()
          }

          ForEach(balances.pools, id: \.source) { pool in
            LabeledContent(pool.source) {
              VStack(alignment: .trailing) {
                Text("\(pool.amount) credits")
                  .monospacedDigit()
                if let expires = pool.expiresAt {
                  Text("expires \(expires)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
              }
            }
          }
        } else {
          Text("Not loaded")
            .foregroundStyle(.secondary)
        }

        Button("Refresh Balances") {
          Task { await core.fetchBalances() }
        }
        .disabled(core.configState?.hasAccount != true)
      }

      // Allocate
      Section("Allocate Credits") {
        HStack {
          TextField("Credits", text: $allocateAmount)
            .textFieldStyle(.roundedBorder)
            .frame(width: 120)

          Button("Allocate") {
            guard let amount = Int64(allocateAmount), amount > 0 else { return }
            Task {
              await core.allocateCredits(amount)
              allocateAmount = ""
              await core.fetchBalances()
            }
          }
          .disabled(
            Int64(allocateAmount) == nil || Int64(allocateAmount)! <= 0
              || core.configState?.hasAccount != true)
        }
      }

      // Prices
      Section("Available Plans") {
        if core.prices.isEmpty {
          Text("No prices loaded")
            .foregroundStyle(.secondary)
        } else {
          ForEach(core.prices, id: \.id) { price in
            VStack(alignment: .leading, spacing: 2) {
              HStack {
                Text(price.productName)
                  .fontWeight(.medium)
                Spacer()
                Text("\(price.amountDisplay)\(price.recurrence)")
                  .foregroundStyle(.secondary)
              }
              Text("\(price.credits) credits")
                .font(.caption)
                .foregroundStyle(.secondary)
              if let desc = price.productDescription {
                Text(desc)
                  .font(.caption)
                  .foregroundStyle(.tertiary)
              }
            }
          }
        }

        Button("Refresh Prices") {
          Task { await core.fetchPrices() }
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
    .navigationTitle("Account")
    .overlay {
      if core.isLoading {
        ProgressView()
          .frame(maxWidth: .infinity, maxHeight: .infinity)
          .background(.ultraThinMaterial)
      }
    }
    .task {
      if core.configState?.hasAccount == true {
        await core.fetchBalances()
      }
      await core.fetchPrices()
    }
  }

  private func showAccount() async {
    // account_show is primarily for CLI; here we just refresh balances
    await core.fetchBalances()
  }
}
