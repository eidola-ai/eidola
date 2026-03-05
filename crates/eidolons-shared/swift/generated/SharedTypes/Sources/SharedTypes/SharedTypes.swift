import Serde


public struct ChatMessage: Hashable {
    @Indirect public var role: SharedTypes.Role
    @Indirect public var content: String

    public init(role: SharedTypes.Role, content: String) {
        self.role = role
        self.content = content
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try self.role.serialize(serializer: serializer)
        try serializer.serialize_str(value: self.content)
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> ChatMessage {
        try deserializer.increase_container_depth()
        let role = try SharedTypes.Role.deserialize(deserializer: deserializer)
        let content = try deserializer.deserialize_str()
        try deserializer.decrease_container_depth()
        return ChatMessage.init(role: role, content: content)
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> ChatMessage {
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
    case perception(SharedTypes.PerceptionRequest)
    case perceptionStreaming(SharedTypes.PerceptionStreamingRequest)

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        switch self {
        case .render(let x):
            try serializer.serialize_variant_index(value: 0)
            try x.serialize(serializer: serializer)
        case .perception(let x):
            try serializer.serialize_variant_index(value: 1)
            try x.serialize(serializer: serializer)
        case .perceptionStreaming(let x):
            try serializer.serialize_variant_index(value: 2)
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
            let x = try SharedTypes.PerceptionRequest.deserialize(deserializer: deserializer)
            try deserializer.decrease_container_depth()
            return .perception(x)
        case 2:
            let x = try SharedTypes.PerceptionStreamingRequest.deserialize(deserializer: deserializer)
            try deserializer.decrease_container_depth()
            return .perceptionStreaming(x)
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
    case submitMessage(String)
    case submitMessageStreaming(String)
    case perceptionChunk(String)
    case perceptionStreamComplete
    case perceptionStreamError(String)

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        switch self {
        case .submitMessage(let x):
            try serializer.serialize_variant_index(value: 0)
            try serializer.serialize_str(value: x)
        case .submitMessageStreaming(let x):
            try serializer.serialize_variant_index(value: 1)
            try serializer.serialize_str(value: x)
        case .perceptionChunk(let x):
            try serializer.serialize_variant_index(value: 2)
            try serializer.serialize_str(value: x)
        case .perceptionStreamComplete:
            try serializer.serialize_variant_index(value: 3)
        case .perceptionStreamError(let x):
            try serializer.serialize_variant_index(value: 4)
            try serializer.serialize_str(value: x)
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
            let x = try deserializer.deserialize_str()
            try deserializer.decrease_container_depth()
            return .submitMessage(x)
        case 1:
            let x = try deserializer.deserialize_str()
            try deserializer.decrease_container_depth()
            return .submitMessageStreaming(x)
        case 2:
            let x = try deserializer.deserialize_str()
            try deserializer.decrease_container_depth()
            return .perceptionChunk(x)
        case 3:
            try deserializer.decrease_container_depth()
            return .perceptionStreamComplete
        case 4:
            let x = try deserializer.deserialize_str()
            try deserializer.decrease_container_depth()
            return .perceptionStreamError(x)
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

public struct PerceptionRequest: Hashable {
    @Indirect public var messages: [SharedTypes.ChatMessage]

    public init(messages: [SharedTypes.ChatMessage]) {
        self.messages = messages
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try serialize_vector_ChatMessage(value: self.messages, serializer: serializer)
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> PerceptionRequest {
        try deserializer.increase_container_depth()
        let messages = try deserialize_vector_ChatMessage(deserializer: deserializer)
        try deserializer.decrease_container_depth()
        return PerceptionRequest.init(messages: messages)
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> PerceptionRequest {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

public struct PerceptionResponse: Hashable {
    @Indirect public var response: String

    public init(response: String) {
        self.response = response
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try serializer.serialize_str(value: self.response)
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> PerceptionResponse {
        try deserializer.increase_container_depth()
        let response = try deserializer.deserialize_str()
        try deserializer.decrease_container_depth()
        return PerceptionResponse.init(response: response)
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> PerceptionResponse {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

public struct PerceptionStreamingRequest: Hashable {
    @Indirect public var messages: [SharedTypes.ChatMessage]

    public init(messages: [SharedTypes.ChatMessage]) {
        self.messages = messages
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try serialize_vector_ChatMessage(value: self.messages, serializer: serializer)
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> PerceptionStreamingRequest {
        try deserializer.increase_container_depth()
        let messages = try deserialize_vector_ChatMessage(deserializer: deserializer)
        try deserializer.decrease_container_depth()
        return PerceptionStreamingRequest.init(messages: messages)
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> PerceptionStreamingRequest {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

indirect public enum PerceptionStreamingResponse: Hashable {
    case chunk(String)
    case done
    case error(String)

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        switch self {
        case .chunk(let x):
            try serializer.serialize_variant_index(value: 0)
            try serializer.serialize_str(value: x)
        case .done:
            try serializer.serialize_variant_index(value: 1)
        case .error(let x):
            try serializer.serialize_variant_index(value: 2)
            try serializer.serialize_str(value: x)
        }
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> PerceptionStreamingResponse {
        let index = try deserializer.deserialize_variant_index()
        try deserializer.increase_container_depth()
        switch index {
        case 0:
            let x = try deserializer.deserialize_str()
            try deserializer.decrease_container_depth()
            return .chunk(x)
        case 1:
            try deserializer.decrease_container_depth()
            return .done
        case 2:
            let x = try deserializer.deserialize_str()
            try deserializer.decrease_container_depth()
            return .error(x)
        default: throw DeserializationError.invalidInput(issue: "Unknown variant index for PerceptionStreamingResponse: \(index)")
        }
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> PerceptionStreamingResponse {
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

indirect public enum Role: Hashable {
    case user
    case assistant

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        switch self {
        case .user:
            try serializer.serialize_variant_index(value: 0)
        case .assistant:
            try serializer.serialize_variant_index(value: 1)
        }
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> Role {
        let index = try deserializer.deserialize_variant_index()
        try deserializer.increase_container_depth()
        switch index {
        case 0:
            try deserializer.decrease_container_depth()
            return .user
        case 1:
            try deserializer.decrease_container_depth()
            return .assistant
        default: throw DeserializationError.invalidInput(issue: "Unknown variant index for Role: \(index)")
        }
    }

    public static func bincodeDeserialize(input: [UInt8]) throws -> Role {
        let deserializer = BincodeDeserializer.init(input: input);
        let obj = try deserialize(deserializer: deserializer)
        if deserializer.get_buffer_offset() < input.count {
            throw DeserializationError.invalidInput(issue: "Some input bytes were not read")
        }
        return obj
    }
}

public struct ViewModel: Hashable {
    @Indirect public var conversation: [SharedTypes.ChatMessage]
    @Indirect public var is_processing: Bool
    @Indirect public var streaming_text: String

    public init(conversation: [SharedTypes.ChatMessage], is_processing: Bool, streaming_text: String) {
        self.conversation = conversation
        self.is_processing = is_processing
        self.streaming_text = streaming_text
    }

    public func serialize<S: Serializer>(serializer: S) throws {
        try serializer.increase_container_depth()
        try serialize_vector_ChatMessage(value: self.conversation, serializer: serializer)
        try serializer.serialize_bool(value: self.is_processing)
        try serializer.serialize_str(value: self.streaming_text)
        try serializer.decrease_container_depth()
    }

    public func bincodeSerialize() throws -> [UInt8] {
        let serializer = BincodeSerializer.init();
        try self.serialize(serializer: serializer)
        return serializer.get_bytes()
    }

    public static func deserialize<D: Deserializer>(deserializer: D) throws -> ViewModel {
        try deserializer.increase_container_depth()
        let conversation = try deserialize_vector_ChatMessage(deserializer: deserializer)
        let is_processing = try deserializer.deserialize_bool()
        let streaming_text = try deserializer.deserialize_str()
        try deserializer.decrease_container_depth()
        return ViewModel.init(conversation: conversation, is_processing: is_processing, streaming_text: streaming_text)
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

func serialize_vector_ChatMessage<S: Serializer>(value: [SharedTypes.ChatMessage], serializer: S) throws {
    try serializer.serialize_len(value: value.count)
    for item in value {
        try item.serialize(serializer: serializer)
    }
}

func deserialize_vector_ChatMessage<D: Deserializer>(deserializer: D) throws -> [SharedTypes.ChatMessage] {
    let length = try deserializer.deserialize_len()
    var obj : [SharedTypes.ChatMessage] = []
    for _ in 0..<length {
        obj.append(try SharedTypes.ChatMessage.deserialize(deserializer: deserializer))
    }
    return obj
}

