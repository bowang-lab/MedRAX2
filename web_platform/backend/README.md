# MedRAX Backend

## SSE Event Stream (Chat)

Endpoint: `POST /api/chats/{chat_id}/stream`

Events are sent as Server-Sent Events. Each event uses the format:

```
event: <event_name>
data: <json>
```

### Event Types

- `message_start`
  - Payload: `{ "messageId": "<user_message_id>" }`

- `content_chunk`
  - Payload: `{ "content": "<assistant_text_chunk>" }`

- `tool_start`
  - Payload: `{ "tool_name": "<tool_id>", "execution_id": "<id>", "message_id": "<assistant_message_id>" }`

- `tool_output`
  - Payload:
    ```json
    {
      "tool_name": "<tool_id>",
      "execution_id": "<id>",
      "message_id": "<assistant_message_id>",
      "result": { "...": "tool output data" },
      "metadata": { "...": "tool metadata" },
      "image_paths": ["/temp/...", "/uploads/..."]
    }
    ```

- `tool_done`
  - Payload: `{ "tool_name": "<tool_id>", "execution_id": "<id>", "message_id": "<assistant_message_id>" }`

- `tool_error`
  - Payload: `{ "tool_name": "<tool_id>", "execution_id": "<id>", "message_id": "<assistant_message_id>", "error": "<error_message>" }`

- `message_done`
  - Payload: `{ "messageId": "<assistant_message_id>" }`

- `error`
  - Payload: `{ "error": "<error_message>" }`

### Notes

- `tool_output` is emitted after the tool finishes and its result is stored.
- `image_paths` are already normalized for frontend display.
