from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GossipMessage(_message.Message):
    __slots__ = ("message", "sender_id", "timestamp", "latency_ms", "round_count")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    ROUND_COUNT_FIELD_NUMBER: _ClassVar[int]
    message: str
    sender_id: str
    timestamp: int
    latency_ms: float
    round_count: int
    def __init__(self, message: _Optional[str] = ..., sender_id: _Optional[str] = ..., timestamp: _Optional[int] = ..., latency_ms: _Optional[float] = ..., round_count: _Optional[int] = ...) -> None: ...

class Acknowledgment(_message.Message):
    __slots__ = ("details",)
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    details: str
    def __init__(self, details: _Optional[str] = ...) -> None: ...

class PenaltyNotification(_message.Message):
    __slots__ = ("message_id", "penalized_neighbor_id", "reporter_node_id", "hop_of_wastage")
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    PENALIZED_NEIGHBOR_ID_FIELD_NUMBER: _ClassVar[int]
    REPORTER_NODE_ID_FIELD_NUMBER: _ClassVar[int]
    HOP_OF_WASTAGE_FIELD_NUMBER: _ClassVar[int]
    message_id: str
    penalized_neighbor_id: str
    reporter_node_id: str
    hop_of_wastage: int
    def __init__(self, message_id: _Optional[str] = ..., penalized_neighbor_id: _Optional[str] = ..., reporter_node_id: _Optional[str] = ..., hop_of_wastage: _Optional[int] = ...) -> None: ...
