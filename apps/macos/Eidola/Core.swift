//
//  Core.swift
//  Eidola
//
//  Observable bridge from SwiftUI to the Rust app core (via UniFFI).
//

import EidolaAppCore
import Foundation

@Observable
@MainActor
public final class Core {
  // MARK: – State

  public private(set) var configState: ConfigState?
  public private(set) var isLoading = false
  public private(set) var errorMessage: String?

  // Account
  public private(set) var prices: [PriceInfo] = []
  public private(set) var balances: BalancesResult?

  // Wallet
  public private(set) var credentials: [CredentialInfo] = []

  // Chat
  public private(set) var models: [ModelInfo] = []
  public private(set) var spaces: [SpaceInfo] = []
  public private(set) var currentSpaceId: String?
  public private(set) var spaceMessages: [SpaceMessage] = []

  // MARK: – Inner core

  private let core: AppCore

  public init() {
    let configDir =
      EidolaAppCore.defaultConfigDir() ?? NSHomeDirectory() + "/Library/Application Support/eidola"
    let dataDir =
      EidolaAppCore.defaultDataDir() ?? NSHomeDirectory() + "/Library/Application Support/eidola"
    core = AppCore(configDir: configDir, dataDir: dataDir)
    refreshConfig()
  }

  // MARK: – Config

  public func refreshConfig() {
    configState = core.configState()
  }

  public func setBaseUrl(_ url: String) throws {
    try core.setBaseUrl(url: url)
    refreshConfig()
  }

  public func setAttestationUrl(_ url: String) throws {
    try core.setAttestationUrl(url: url)
    refreshConfig()
  }

  public func setAccountCredentials(id: String, secret: String) throws {
    try core.setAccountCredentials(id: id, secret: secret)
    refreshConfig()
  }

  public func resetAccount() throws {
    try core.resetAccount()
    refreshConfig()
  }

  // MARK: – Account

  public func createAccount() async {
    isLoading = true
    errorMessage = nil
    do {
      _ = try await core.accountCreate()
      refreshConfig()
    } catch {
      errorMessage = error.localizedDescription
    }
    isLoading = false
  }

  public func fetchPrices() async {
    isLoading = true
    errorMessage = nil
    do {
      prices = try await core.accountPrices()
    } catch {
      errorMessage = error.localizedDescription
    }
    isLoading = false
  }

  public func checkout(priceId: String) async -> String? {
    isLoading = true
    errorMessage = nil
    do {
      let url = try await core.accountCheckout(priceId: priceId)
      isLoading = false
      return url
    } catch {
      errorMessage = error.localizedDescription
      isLoading = false
      return nil
    }
  }

  public func fetchBalances() async {
    isLoading = true
    errorMessage = nil
    do {
      balances = try await core.accountBalances()
    } catch {
      errorMessage = error.localizedDescription
    }
    isLoading = false
  }

  public func allocateCredits(_ credits: Int64) async {
    isLoading = true
    errorMessage = nil
    do {
      _ = try await core.accountAllocate(credits: credits)
      credentials = try await core.walletCredentials()
    } catch {
      errorMessage = error.localizedDescription
    }
    isLoading = false
  }

  // MARK: – Wallet

  public func fetchCredentials() async {
    isLoading = true
    errorMessage = nil
    do {
      credentials = try await core.walletCredentials()
    } catch {
      errorMessage = error.localizedDescription
    }
    isLoading = false
  }

  // MARK: – Models

  public func fetchModels() async {
    isLoading = true
    errorMessage = nil
    do {
      models = try await core.availableModels()
    } catch {
      errorMessage = error.localizedDescription
    }
    isLoading = false
  }

  // MARK: – Spaces

  public func fetchSpaces() async {
    do {
      spaces = try await core.listSpaces()
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  public func selectSpace(_ id: String) async {
    currentSpaceId = id
    do {
      spaceMessages = try await core.getSpaceMessages(spaceId: id)
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  public func newChat() {
    currentSpaceId = nil
    spaceMessages = []
  }

  public func archiveSpace(_ id: String) async {
    do {
      _ = try await core.archiveSpace(spaceId: id)
      if currentSpaceId == id {
        newChat()
      }
      await fetchSpaces()
    } catch {
      errorMessage = error.localizedDescription
    }
  }

  // MARK: – Chat

  public func sendMessage(_ prompt: String, model: String) async {
    // Optimistically show the user message
    spaceMessages.append(SpaceMessage(role: "user", content: prompt))

    isLoading = true
    errorMessage = nil
    do {
      let result = try await core.chat(prompt: prompt, model: model, spaceId: currentSpaceId)
      currentSpaceId = result.spaceId

      // Reload messages from DB to get the authoritative state
      spaceMessages = try await core.getSpaceMessages(spaceId: result.spaceId)

      // Refresh space list (new space may have been created)
      await fetchSpaces()
    } catch {
      errorMessage = error.localizedDescription
      spaceMessages.append(SpaceMessage(role: "error", content: error.localizedDescription))
    }
    isLoading = false
  }
}
