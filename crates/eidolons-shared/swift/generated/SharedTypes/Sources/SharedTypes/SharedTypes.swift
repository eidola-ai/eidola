import Serde


public struct BalancePoolViewModel: Hashable {
    @Indirect public var amount: Int64
    @Indirect public var source: String
    @Indirect public var expires_at: String?

    public init(amount: Int64, source: String, expires_at: String?) {
        self.amount = amount
        self.source = source
        self.expires_at = expires_at
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try serializer.serialize_i64(value: self.amount)
        try serializer.serialize_str(value: self.source)
        try serialize_option_str(value: self.expires_at, serializer: serializer)
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> BalancePoolViewModel {
        try deserializer.increase_container_depth()
        let amount = try deserializer.deserialize_i64()
        let source = try deserializer.deserialize_str()
        let expires_at = try deserialize_option_str(deserializer: deserializer)
        try deserializer.decrease_container_depth()
        return BalancePoolViewModel.init(amount: amount, source: source, expires_at: expires_at)
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> BalancePoolViewModel {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

indirect public enum Effect: Hashable {
    case render(SharedTypes.RenderOperation)
    case http(SharedTypes.HttpRequest)

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        switch self {
        case .render(let x):
            try serializer.serialize_variant_index(value: 0)
            try x.serialize(serializer: serializer)
        case .http(let x):
            try serializer.serialize_variant_index(value: 1)
            try x.serialize(serializer: serializer)
        }
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> Effect {
        let index = try deserializer.deserialize_variant_index()
        try deserializer.increase_container_depth()
        switch index {
        case 0:
            let x = try SharedTypes.RenderOperation.deserialize(deserializer: deserializer)
            try deserializer.decrease_container_depth()
            return .render(x)
        case 1:
            let x = try SharedTypes.HttpRequest.deserialize(deserializer: deserializer)
            try deserializer.decrease_container_depth()
            return .http(x)
        default: throw DeserializationError.invalidInput(issue: "Unknown variant index for Effect: \(index)")
        }
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> Effect {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

indirect public enum Event: Hashable {
    case init(base_url: String, account_id: String?, account_secret: String?)
    case getAccount
    case createAccount
    case getPrices
    case checkout(price_id: String)
    case getBalances

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        switch self {
        case .init(let base_url, let account_id, let account_secret):
            try serializer.serialize_variant_index(value: 0)
            try serializer.serialize_str(value: base_url)
            try serialize_option_str(value: account_id, serializer: serializer)
            try serialize_option_str(value: account_secret, serializer: serializer)
        case .getAccount:
            try serializer.serialize_variant_index(value: 1)
        case .createAccount:
            try serializer.serialize_variant_index(value: 2)
        case .getPrices:
            try serializer.serialize_variant_index(value: 3)
        case .checkout(let price_id):
            try serializer.serialize_variant_index(value: 4)
            try serializer.serialize_str(value: price_id)
        case .getBalances:
            try serializer.serialize_variant_index(value: 5)
        }
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> Event {
        let index = try deserializer.deserialize_variant_index()
        try deserializer.increase_container_depth()
        switch index {
        case 0:
            let base_url = try deserializer.deserialize_str()
            let account_id = try deserialize_option_str(deserializer: deserializer)
            let account_secret = try deserialize_option_str(deserializer: deserializer)
            try deserializer.decrease_container_depth()
            return .init(base_url: base_url, account_id: account_id, account_secret: account_secret)
        case 1:
            try deserializer.decrease_container_depth()
            return .getAccount
        case 2:
            try deserializer.decrease_container_depth()
            return .createAccount
        case 3:
            try deserializer.decrease_container_depth()
            return .getPrices
        case 4:
            let price_id = try deserializer.deserialize_str()
            try deserializer.decrease_container_depth()
            return .checkout(price_id: price_id)
        case 5:
            try deserializer.decrease_container_depth()
            return .getBalances
        default: throw DeserializationError.invalidInput(issue: "Unknown variant index for Event: \(index)")
        }
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> Event {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

indirect public enum HttpError: Hashable {
    case url(String)
    case io(String)
    case timeout

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        switch self {
        case .url(let x):
            try serializer.serialize_variant_index(value: 0)
            try serializer.serialize_str(value: x)
        case .io(let x):
            try serializer.serialize_variant_index(value: 1)
            try serializer.serialize_str(value: x)
        case .timeout:
            try serializer.serialize_variant_index(value: 2)
        }
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> HttpError {
        let index = try deserializer.deserialize_variant_index()
        try deserializer.increase_container_depth()
        switch index {
        case 0:
            let x = try deserializer.deserialize_str()
            try deserializer.decrease_container_depth()
            return .url(x)
        case 1:
            let x = try deserializer.deserialize_str()
            try deserializer.decrease_container_depth()
            return .io(x)
        case 2:
            try deserializer.decrease_container_depth()
            return .timeout
        default: throw DeserializationError.invalidInput(issue: "Unknown variant index for HttpError: \(index)")
        }
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> HttpError {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

public struct HttpHeader: Hashable {
    @Indirect public var name: String
    @Indirect public var value: String

    public init(name: String, value: String) {
        self.name = name
        self.value = value
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try serializer.serialize_str(value: self.name)
        try serializer.serialize_str(value: self.value)
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> HttpHeader {
        try deserializer.increase_container_depth()
        let name = try deserializer.deserialize_str()
        let value = try deserializer.deserialize_str()
        try deserializer.decrease_container_depth()
        return HttpHeader.init(name: name, value: value)
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> HttpHeader {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

public struct HttpRequest: Hashable {
    @Indirect public var method: String
    @Indirect public var url: String
    @Indirect public var headers: [SharedTypes.HttpHeader]
    @Indirect public var body: [UInt8]

    public init(method: String, url: String, headers: [SharedTypes.HttpHeader], body: [UInt8]) {
        self.method = method
        self.url = url
        self.headers = headers
        self.body = body
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try serializer.serialize_str(value: self.method)
        try serializer.serialize_str(value: self.url)
        try serialize_vector_HttpHeader(value: self.headers, serializer: serializer)
        try serializer.serialize_bytes(value: self.body)
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> HttpRequest {
        try deserializer.increase_container_depth()
        let method = try deserializer.deserialize_str()
        let url = try deserializer.deserialize_str()
        let headers = try deserialize_vector_HttpHeader(deserializer: deserializer)
        let body = try deserializer.deserialize_bytes()
        try deserializer.decrease_container_depth()
        return HttpRequest.init(method: method, url: url, headers: headers, body: body)
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> HttpRequest {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

public struct HttpResponse: Hashable {
    @Indirect public var status: UInt16
    @Indirect public var headers: [SharedTypes.HttpHeader]
    @Indirect public var body: [UInt8]

    public init(status: UInt16, headers: [SharedTypes.HttpHeader], body: [UInt8]) {
        self.status = status
        self.headers = headers
        self.body = body
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try serializer.serialize_u16(value: self.status)
        try serialize_vector_HttpHeader(value: self.headers, serializer: serializer)
        try serializer.serialize_bytes(value: self.body)
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> HttpResponse {
        try deserializer.increase_container_depth()
        let status = try deserializer.deserialize_u16()
        let headers = try deserialize_vector_HttpHeader(deserializer: deserializer)
        let body = try deserializer.deserialize_bytes()
        try deserializer.decrease_container_depth()
        return HttpResponse.init(status: status, headers: headers, body: body)
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> HttpResponse {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

indirect public enum HttpResult: Hashable {
    case ok(SharedTypes.HttpResponse)
    case err(SharedTypes.HttpError)

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        switch self {
        case .ok(let x):
            try serializer.serialize_variant_index(value: 0)
            try x.serialize(serializer: serializer)
        case .err(let x):
            try serializer.serialize_variant_index(value: 1)
            try x.serialize(serializer: serializer)
        }
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> HttpResult {
        let index = try deserializer.deserialize_variant_index()
        try deserializer.increase_container_depth()
        switch index {
        case 0:
            let x = try SharedTypes.HttpResponse.deserialize(deserializer: deserializer)
            try deserializer.decrease_container_depth()
            return .ok(x)
        case 1:
            let x = try SharedTypes.HttpError.deserialize(deserializer: deserializer)
            try deserializer.decrease_container_depth()
            return .err(x)
        default: throw DeserializationError.invalidInput(issue: "Unknown variant index for HttpResult: \(index)")
        }
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> HttpResult {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

public struct PriceViewModel: Hashable {
    @Indirect public var id: String
    @Indirect public var product_name: String
    @Indirect public var product_description: String?
    @Indirect public var unit_amount: Int64?
    @Indirect public var currency: String
    @Indirect public var recurring_interval: String?
    @Indirect public var recurring_interval_count: Int64?
    @Indirect public var credits: Int64

    public init(id: String, product_name: String, product_description: String?, unit_amount: Int64?, currency: String, recurring_interval: String?, recurring_interval_count: Int64?, credits: Int64) {
        self.id = id
        self.product_name = product_name
        self.product_description = product_description
        self.unit_amount = unit_amount
        self.currency = currency
        self.recurring_interval = recurring_interval
        self.recurring_interval_count = recurring_interval_count
        self.credits = credits
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try serializer.serialize_str(value: self.id)
        try serializer.serialize_str(value: self.product_name)
        try serialize_option_str(value: self.product_description, serializer: serializer)
        try serialize_option_i64(value: self.unit_amount, serializer: serializer)
        try serializer.serialize_str(value: self.currency)
        try serialize_option_str(value: self.recurring_interval, serializer: serializer)
        try serialize_option_i64(value: self.recurring_interval_count, serializer: serializer)
        try serializer.serialize_i64(value: self.credits)
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> PriceViewModel {
        try deserializer.increase_container_depth()
        let id = try deserializer.deserialize_str()
        let product_name = try deserializer.deserialize_str()
        let product_description = try deserialize_option_str(deserializer: deserializer)
        let unit_amount = try deserialize_option_i64(deserializer: deserializer)
        let currency = try deserializer.deserialize_str()
        let recurring_interval = try deserialize_option_str(deserializer: deserializer)
        let recurring_interval_count = try deserialize_option_i64(deserializer: deserializer)
        let credits = try deserializer.deserialize_i64()
        try deserializer.decrease_container_depth()
        return PriceViewModel.init(id: id, product_name: product_name, product_description: product_description, unit_amount: unit_amount, currency: currency, recurring_interval: recurring_interval, recurring_interval_count: recurring_interval_count, credits: credits)
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> PriceViewModel {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

public struct RenderOperation: Hashable {

    public init() {
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> RenderOperation {
        try deserializer.increase_container_depth()
        try deserializer.decrease_container_depth()
        return RenderOperation.init()
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> RenderOperation {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

public struct Request: Hashable {
    @Indirect public var id: UInt32
    @Indirect public var effect: SharedTypes.Effect

    public init(id: UInt32, effect: SharedTypes.Effect) {
        self.id = id
        self.effect = effect
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try serializer.serialize_u32(value: self.id)
        try self.effect.serialize(serializer: serializer)
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> Request {
        try deserializer.increase_container_depth()
        let id = try deserializer.deserialize_u32()
        let effect = try SharedTypes.Effect.deserialize(deserializer: deserializer)
        try deserializer.decrease_container_depth()
        return Request.init(id: id, effect: effect)
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> Request {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

indirect public enum Screen: Hashable {
    case idle
    case loading
    case account
    case accountCreated
    case prices
    case checkout
    case balances
    case error

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        switch self {
        case .idle:
            try serializer.serialize_variant_index(value: 0)
        case .loading:
            try serializer.serialize_variant_index(value: 1)
        case .account:
            try serializer.serialize_variant_index(value: 2)
        case .accountCreated:
            try serializer.serialize_variant_index(value: 3)
        case .prices:
            try serializer.serialize_variant_index(value: 4)
        case .checkout:
            try serializer.serialize_variant_index(value: 5)
        case .balances:
            try serializer.serialize_variant_index(value: 6)
        case .error:
            try serializer.serialize_variant_index(value: 7)
        }
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> Screen {
        let index = try deserializer.deserialize_variant_index()
        try deserializer.increase_container_depth()
        switch index {
        case 0:
            try deserializer.decrease_container_depth()
            return .idle
        case 1:
            try deserializer.decrease_container_depth()
            return .loading
        case 2:
            try deserializer.decrease_container_depth()
            return .account
        case 3:
            try deserializer.decrease_container_depth()
            return .accountCreated
        case 4:
            try deserializer.decrease_container_depth()
            return .prices
        case 5:
            try deserializer.decrease_container_depth()
            return .checkout
        case 6:
            try deserializer.decrease_container_depth()
            return .balances
        case 7:
            try deserializer.decrease_container_depth()
            return .error
        default: throw DeserializationError.invalidInput(issue: "Unknown variant index for Screen: \(index)")
        }
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> Screen {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

public struct ViewModel: Hashable {
    @Indirect public var screen: SharedTypes.Screen
    @Indirect public var error: String?
    @Indirect public var account_id: String?
    @Indirect public var account_stripe_customer_id: String?
    @Indirect public var account_created_at: String?
    @Indirect public var created_account_id: String?
    @Indirect public var created_account_secret: String?
    @Indirect public var created_account_created_at: String?
    @Indirect public var prices: [SharedTypes.PriceViewModel]
    @Indirect public var checkout_url: String?
    @Indirect public var balances_available: Int64?
    @Indirect public var balances_pools: [SharedTypes.BalancePoolViewModel]

    public init(screen: SharedTypes.Screen, error: String?, account_id: String?, account_stripe_customer_id: String?, account_created_at: String?, created_account_id: String?, created_account_secret: String?, created_account_created_at: String?, prices: [SharedTypes.PriceViewModel], checkout_url: String?, balances_available: Int64?, balances_pools: [SharedTypes.BalancePoolViewModel]) {
        self.screen = screen
        self.error = error
        self.account_id = account_id
        self.account_stripe_customer_id = account_stripe_customer_id
        self.account_created_at = account_created_at
        self.created_account_id = created_account_id
        self.created_account_secret = created_account_secret
        self.created_account_created_at = created_account_created_at
        self.prices = prices
        self.checkout_url = checkout_url
        self.balances_available = balances_available
        self.balances_pools = balances_pools
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try self.screen.serialize(serializer: serializer)
        try serialize_option_str(value: self.error, serializer: serializer)
        try serialize_option_str(value: self.account_id, serializer: serializer)
        try serialize_option_str(value: self.account_stripe_customer_id, serializer: serializer)
        try serialize_option_str(value: self.account_created_at, serializer: serializer)
        try serialize_option_str(value: self.created_account_id, serializer: serializer)
        try serialize_option_str(value: self.created_account_secret, serializer: serializer)
        try serialize_option_str(value: self.created_account_created_at, serializer: serializer)
        try serialize_vector_PriceViewModel(value: self.prices, serializer: serializer)
        try serialize_option_str(value: self.checkout_url, serializer: serializer)
        try serialize_option_i64(value: self.balances_available, serializer: serializer)
        try serialize_vector_BalancePoolViewModel(value: self.balances_pools, serializer: serializer)
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> ViewModel {
        try deserializer.increase_container_depth()
        let screen = try SharedTypes.Screen.deserialize(deserializer: deserializer)
        let error = try deserialize_option_str(deserializer: deserializer)
        let account_id = try deserialize_option_str(deserializer: deserializer)
        let account_stripe_customer_id = try deserialize_option_str(deserializer: deserializer)
        let account_created_at = try deserialize_option_str(deserializer: deserializer)
        let created_account_id = try deserialize_option_str(deserializer: deserializer)
        let created_account_secret = try deserialize_option_str(deserializer: deserializer)
        let created_account_created_at = try deserialize_option_str(deserializer: deserializer)
        let prices = try deserialize_vector_PriceViewModel(deserializer: deserializer)
        let checkout_url = try deserialize_option_str(deserializer: deserializer)
        let balances_available = try deserialize_option_i64(deserializer: deserializer)
        let balances_pools = try deserialize_vector_BalancePoolViewModel(deserializer: deserializer)
        try deserializer.decrease_container_depth()
        return ViewModel.init(screen: screen, error: error, account_id: account_id, account_stripe_customer_id: account_stripe_customer_id, account_created_at: account_created_at, created_account_id: created_account_id, created_account_secret: created_account_secret, created_account_created_at: created_account_created_at, prices: prices, checkout_url: checkout_url, balances_available: balances_available, balances_pools: balances_pools)
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> ViewModel {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

func serialize_option_i64<S: Serializer>(value: Int64?, serializer: S) throws {
    if let value = value {
        try serializer.serialize_option_tag(value: true)
        try serializer.serialize_i64(value: value)
    } else {
        try serializer.serialize_option_tag(value: false)
    }
}

func deserialize_option_i64<D: Deserializer>(deserializer: D) throws -> Int64? {
    let tag = try deserializer.deserialize_option_tag()
    if tag {
        return try deserializer.deserialize_i64()
    } else {
        return nil
    }
}

func serialize_option_str<S: Serializer>(value: String?, serializer: S) throws {
    if let value = value {
        try serializer.serialize_option_tag(value: true)
        try serializer.serialize_str(value: value)
    } else {
        try serializer.serialize_option_tag(value: false)
    }
}

func deserialize_option_str<D: Deserializer>(deserializer: D) throws -> String? {
    let tag = try deserializer.deserialize_option_tag()
    if tag {
        return try deserializer.deserialize_str()
    } else {
        return nil
    }
}

func serialize_vector_BalancePoolViewModel<S: Serializer>(value: [SharedTypes.BalancePoolViewModel], serializer: S) throws {
    try serializer.serialize_len(value: value.count)
    for item in value {
        try item.serialize(serializer: serializer)
    }
}

func deserialize_vector_BalancePoolViewModel<D: Deserializer>(deserializer: D) throws -> [SharedTypes.BalancePoolViewModel] {
    let length = try deserializer.deserialize_len()
    var obj : [SharedTypes.BalancePoolViewModel] = []
    for _ in 0..<length {
        obj.append(try SharedTypes.BalancePoolViewModel.deserialize(deserializer: deserializer))
    }
    return obj
}

func serialize_vector_HttpHeader<S: Serializer>(value: [SharedTypes.HttpHeader], serializer: S) throws {
    try serializer.serialize_len(value: value.count)
    for item in value {
        try item.serialize(serializer: serializer)
    }
}

func deserialize_vector_HttpHeader<D: Deserializer>(deserializer: D) throws -> [SharedTypes.HttpHeader] {
    let length = try deserializer.deserialize_len()
    var obj : [SharedTypes.HttpHeader] = []
    for _ in 0..<length {
        obj.append(try SharedTypes.HttpHeader.deserialize(deserializer: deserializer))
    }
    return obj
}

func serialize_vector_PriceViewModel<S: Serializer>(value: [SharedTypes.PriceViewModel], serializer: S) throws {
    try serializer.serialize_len(value: value.count)
    for item in value {
        try item.serialize(serializer: serializer)
    }
}

func deserialize_vector_PriceViewModel<D: Deserializer>(deserializer: D) throws -> [SharedTypes.PriceViewModel] {
    let length = try deserializer.deserialize_len()
    var obj : [SharedTypes.PriceViewModel] = []
    for _ in 0..<length {
        obj.append(try SharedTypes.PriceViewModel.deserialize(deserializer: deserializer))
    }
    return obj
}

