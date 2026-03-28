// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

func TestConvertResponse_WithReasoningContent(t *testing.T) {
	resp := &chatCompletionsResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "minimax-01",
		Choices: []choice{
			{
				Index: 0,
				Message: &responseMessage{
					Role:             "assistant",
					Content:          "Final answer",
					ReasoningContent: "Thinking process...",
				},
				FinishReason: "stop",
			},
		},
		Usage: &usage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}

	m := &openAIModel{name: "test-model"}
	llmResp, err := m.convertResponse(resp)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if llmResp.ThinkingContent != "Thinking process..." {
		t.Errorf("ThinkingContent = %q, want %q", llmResp.ThinkingContent, "Thinking process...")
	}

	if llmResp.Content == nil || len(llmResp.Content.Parts) == 0 {
		t.Fatal("Content should not be nil or empty")
	}

	if llmResp.Content.Parts[0].Text != "Final answer" {
		t.Errorf("Content text = %q, want %q", llmResp.Content.Parts[0].Text, "Final answer")
	}

	if llmResp.UsageMetadata == nil {
		t.Fatal("UsageMetadata should not be nil")
	}
	if llmResp.UsageMetadata.PromptTokenCount != 10 {
		t.Errorf("PromptTokenCount = %d, want %d", llmResp.UsageMetadata.PromptTokenCount, 10)
	}
}

func TestConvertResponse_NoChoices(t *testing.T) {
	resp := &chatCompletionsResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "minimax-01",
		Choices: []choice{},
	}

	m := &openAIModel{name: "test-model"}
	llmResp, err := m.convertResponse(resp)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if llmResp.ErrorCode != "NO_CHOICES" {
		t.Errorf("ErrorCode = %q, want %q", llmResp.ErrorCode, "NO_CHOICES")
	}
}

func TestConvertResponse_NilMessage(t *testing.T) {
	resp := &chatCompletionsResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "minimax-01",
		Choices: []choice{
			{
				Index:        0,
				Message:      nil,
				FinishReason: "stop",
			},
		},
	}

	m := &openAIModel{name: "test-model"}
	llmResp, err := m.convertResponse(resp)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if llmResp.ErrorCode != "NO_MESSAGE" {
		t.Errorf("ErrorCode = %q, want %q", llmResp.ErrorCode, "NO_MESSAGE")
	}
}

func TestConvertStreamChoice_WithReasoning(t *testing.T) {
	c := &choice{
		Delta: &responseMessage{
			Content:          "Part of answer",
			ReasoningContent: "Reasoning...",
		},
		FinishReason: "null",
	}

	m := &openAIModel{name: "test-model"}
	llmResp := m.convertStreamChoice(c, nil)

	if llmResp.ThinkingContent != "Reasoning..." {
		t.Errorf("ThinkingContent = %q, want %q", llmResp.ThinkingContent, "Reasoning...")
	}

	if !llmResp.Partial {
		t.Error("Partial should be true for non-final chunk")
	}

	if llmResp.TurnComplete {
		t.Error("TurnComplete should be false for non-final chunk")
	}
}

func TestConvertStreamChoice_Final(t *testing.T) {
	c := &choice{
		Delta: &responseMessage{
			Content:          "Final answer",
			ReasoningContent: "Final reasoning",
		},
		FinishReason: "stop",
	}

	m := &openAIModel{name: "test-model"}
	llmResp := m.convertStreamChoice(c, nil)

	if llmResp.ThinkingContent != "Final reasoning" {
		t.Errorf("ThinkingContent = %q, want %q", llmResp.ThinkingContent, "Final reasoning")
	}

	if llmResp.Partial {
		t.Error("Partial should be false for final chunk")
	}

	if !llmResp.TurnComplete {
		t.Error("TurnComplete should be true for final chunk")
	}
}

func TestConvertStreamChoice_NilDelta(t *testing.T) {
	c := &choice{
		Delta:        nil,
		FinishReason: "stop",
	}

	m := &openAIModel{name: "test-model"}
	llmResp := m.convertStreamChoice(c, nil)

	if llmResp.ThinkingContent != "" {
		t.Errorf("ThinkingContent = %q, want empty string", llmResp.ThinkingContent)
	}

	if !llmResp.TurnComplete {
		t.Error("TurnComplete should be true for final chunk")
	}
}

func TestGetReasoningContent(t *testing.T) {
	tests := []struct {
		name  string
		delta *responseMessage
		want  string
	}{
		{
			name:  "nil delta",
			delta: nil,
			want:  "",
		},
		{
			name:  "empty reasoning",
			delta: &responseMessage{},
			want:  "",
		},
		{
			name:  "with reasoning",
			delta: &responseMessage{ReasoningContent: "thinking..."},
			want:  "thinking...",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getReasoningContent(tt.delta)
			if got != tt.want {
				t.Errorf("getReasoningContent() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestConvertRequest_WithThinking(t *testing.T) {
	req := &model.LLMRequest{
		Model: "minimax-01",
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Hello"},
				},
			},
		},
		Tools: map[string]any{},
	}

	m := &openAIModel{name: "test-model"}
	openAIReq, err := m.convertRequest(req, false)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if openAIReq.Model != "minimax-01" {
		t.Errorf("Model = %q, want %q", openAIReq.Model, "minimax-01")
	}

	if len(openAIReq.Messages) != 1 {
		t.Fatalf("Messages length = %d, want 1", len(openAIReq.Messages))
	}

	if openAIReq.Messages[0]["role"] != "user" {
		t.Errorf("Message role = %q, want %q", openAIReq.Messages[0]["role"], "user")
	}
}

func TestParseStream_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal server error"))
	}))
	defer server.Close()

	m := &openAIModel{
		name:       "test-model",
		apiKey:     "test-key",
		baseURL:   server.URL,
		httpClient: server.Client(),
	}

	req := &model.LLMRequest{
		Model: "test-model",
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
		},
	}

	var lastErr error
	for resp, err := range m.GenerateContent(nil, req, true) {
		if err != nil {
			lastErr = err
			break
		}
		if resp != nil && resp.ErrorCode != "" {
			lastErr = nil
			break
		}
	}

	if lastErr == nil {
		t.Fatal("expected error for server error response")
	}
}

func TestParseStream_WithReasoningContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)

		// First chunk with reasoning
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking...\"},\"finish_reason\":\"null\"}]}\n\n"))
		flusher.Flush()

		// Second chunk with content
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"null\"}]}\n\n"))
		flusher.Flush()

		// Final chunk
		w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
		flusher.Flush()

		w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer server.Close()

	m := &openAIModel{
		name:       "test-model",
		apiKey:     "test-key",
		baseURL:   server.URL,
		httpClient: server.Client(),
	}

	req := &model.LLMRequest{
		Model: "test-model",
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
		},
	}

	var lastResp *model.LLMResponse
	count := 0
	var lastErr error
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			lastErr = err
			break
		}
		count++
		lastResp = resp
	}

	if lastErr != nil {
		t.Fatalf("unexpected error: %v", lastErr)
	}

	if count < 2 {
		t.Errorf("expected at least 2 responses, got %d", count)
	}

	if lastResp == nil {
		t.Fatal("expected at least one response")
	}

	if !lastResp.TurnComplete {
		t.Error("expected TurnComplete to be true on final chunk")
	}
}

func TestResponseMessage_JSON(t *testing.T) {
	jsonStr := `{
		"role": "assistant",
		"content": "Hello",
		"reasoning_content": "I am thinking..."
	}`

	var msg responseMessage
	if err := json.Unmarshal([]byte(jsonStr), &msg); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if msg.Content != "Hello" {
		t.Errorf("Content = %q, want %q", msg.Content, "Hello")
	}

	if msg.ReasoningContent != "I am thinking..." {
		t.Errorf("ReasoningContent = %q, want %q", msg.ReasoningContent, "I am thinking...")
	}
}

func TestChatCompletionsRequest_Thinking(t *testing.T) {
	thinking := map[string]interface{}{
		"type":          "enabled",
		"budget_tokens": 1000,
	}

	openAIReq := &chatCompletionsRequest{
		Model:    "minimax-01",
		Messages: []map[string]interface{}{{"role": "user", "content": "hi"}},
		Thinking: thinking,
	}

	data, err := json.Marshal(openAIReq)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	if !strings.Contains(string(data), "thinking") {
		t.Error("json should contain thinking field")
	}
}
