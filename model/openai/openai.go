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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

type Config struct {
	ModelName  string
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
}

type Option func(*Config)

func WithAPIKey(apiKey string) Option {
	return func(c *Config) {
		c.APIKey = apiKey
	}
}

func WithBaseURL(baseURL string) Option {
	return func(c *Config) {
		c.BaseURL = strings.TrimSuffix(baseURL, "/")
	}
}

func WithHTTPClient(client *http.Client) Option {
	return func(c *Config) {
		c.HTTPClient = client
	}
}

type openAIModel struct {
	name       string
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

func NewModel(ctx context.Context, modelName string, opts ...Option) (model.LLM, error) {
	cfg := &Config{
		ModelName: modelName,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	if cfg.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("base URL is required")
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	return &openAIModel{
		name:       cfg.ModelName,
		apiKey:     cfg.APIKey,
		baseURL:    cfg.BaseURL,
		httpClient: httpClient,
	}, nil
}

func (m *openAIModel) Name() string {
	return m.name
}

func (m *openAIModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return m.generateContentStream(ctx, req)
	}
	return m.generateContentNonStream(ctx, req)
}

func (m *openAIModel) generateContentNonStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		openAIReq, err := m.convertRequest(req, false)
		if err != nil {
			yield(nil, err)
			return
		}

		body, err := json.Marshal(openAIReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to marshal request: %w", err))
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/chat/completions", bytes.NewReader(body))
		if err != nil {
			yield(nil, fmt.Errorf("failed to create request: %w", err))
			return
		}

		m.addHeaders(httpReq)

		resp, err := m.httpClient.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to call API: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			yield(nil, fmt.Errorf("API error: status %d, body: %s", resp.StatusCode, string(respBody)))
			return
		}

		var openAIResp chatCompletionsResponse
		if err := json.NewDecoder(resp.Body).Decode(&openAIResp); err != nil {
			yield(nil, fmt.Errorf("failed to decode response: %w", err))
			return
		}

		llmResp, err := m.convertResponse(&openAIResp)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(llmResp, nil)
	}
}

func (m *openAIModel) generateContentStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		openAIReq, err := m.convertRequest(req, true)
		if err != nil {
			yield(nil, err)
			return
		}

		body, err := json.Marshal(openAIReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to marshal request: %w", err))
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/chat/completions", bytes.NewReader(body))
		if err != nil {
			yield(nil, fmt.Errorf("failed to create request: %w", err))
			return
		}

		m.addHeaders(httpReq)

		resp, err := m.httpClient.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to call API: %w", err))
			return
		}

		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			yield(nil, fmt.Errorf("API error: status %d, body: %s", resp.StatusCode, string(respBody)))
			return
		}

		if !m.parseStream(resp.Body, yield) {
			return
		}
	}
}

func (m *openAIModel) addHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)
	req.Header.Set("User-Agent", "google-adk/go")
}

type chatCompletionsRequest struct {
	Model    string                   `json:"model"`
	Messages []map[string]interface{} `json:"messages"`
	Tools    []map[string]interface{} `json:"tools,omitempty"`
	Stream   bool                     `json:"stream,omitempty"`
	Thinking map[string]interface{}   `json:"thinking,omitempty"`
}

type chatCompletionsResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []choice `json:"choices"`
	Usage   *usage   `json:"usage,omitempty"`
}

type choice struct {
	Index        int              `json:"index"`
	Message      *responseMessage `json:"message,omitempty"`
	Delta        *responseMessage `json:"delta,omitempty"`
	FinishReason string           `json:"finish_reason"`
}

type responseMessage struct {
	Role             string     `json:"role,omitempty"`
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []toolCall `json:"tool_calls,omitempty"`
}

type toolCall struct {
	Index    int      `json:"index,omitempty"`
	ID       string   `json:"id,omitempty"`
	Type     string   `json:"type,omitempty"`
	Function function `json:"function,omitempty"`
}

type function struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (m *openAIModel) convertRequest(req *model.LLMRequest, stream bool) (*chatCompletionsRequest, error) {
	messages, err := m.convertContents(req.Contents)
	if err != nil {
		return nil, err
	}

	openAIReq := &chatCompletionsRequest{
		Model:    m.modelName(req),
		Messages: messages,
		Stream:   stream,
	}

	if len(req.Tools) > 0 {
		tools, err := m.convertTools(req.Tools)
		if err != nil {
			return nil, err
		}
		openAIReq.Tools = tools
	}

	return openAIReq, nil
}

func (m *openAIModel) modelName(req *model.LLMRequest) string {
	if req.Model != "" {
		return req.Model
	}
	return m.name
}

func (m *openAIModel) convertContents(contents []*genai.Content) ([]map[string]interface{}, error) {
	var messages []map[string]interface{}
	for _, content := range contents {
		if content == nil {
			continue
		}
		msg, err := m.convertContent(content)
		if err != nil {
			return nil, err
		}
		if msg != nil {
			messages = append(messages, msg)
		}
	}
	return messages, nil
}

func (m *openAIModel) convertContent(content *genai.Content) (map[string]interface{}, error) {
	role := m.convertRole(content.Role)
	if role == "" {
		return nil, nil
	}

	contents, err := m.convertParts(content.Parts)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"role":    role,
		"content": contents,
	}, nil
}

func (m *openAIModel) convertRole(role string) string {
	switch role {
	case "user":
		return "user"
	case "model":
		return "assistant"
	case "system":
		return "system"
	default:
		return role
	}
}

func (m *openAIModel) convertParts(parts []*genai.Part) (interface{}, error) {
	if len(parts) == 0 {
		return nil, nil
	}

	var texts []string
	var imageURLs []map[string]interface{}
	var toolCalls []map[string]interface{}

	for _, part := range parts {
		if part == nil {
			continue
		}

		if part.Text != "" {
			texts = append(texts, part.Text)
		}

		if part.InlineData != nil {
			blob := part.InlineData
			imageURLs = append(imageURLs, map[string]interface{}{
				"type": "image_url",
				"image_url": map[string]interface{}{
					"url":    fmt.Sprintf("data:%s;base64,%s", blob.MIMEType, string(blob.Data)),
					"detail": "high",
				},
			})
		}

		if part.FileData != nil {
			fileData := part.FileData
			imageURLs = append(imageURLs, map[string]interface{}{
				"type": "image_url",
				"image_url": map[string]interface{}{
					"url": fileData.FileURI,
				},
			})
		}

		if part.FunctionCall != nil {
			fc := part.FunctionCall
			argsJSON, err := json.Marshal(fc.Args)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function arguments: %w", err)
			}
			toolCalls = append(toolCalls, map[string]interface{}{
				"id":   fc.ID,
				"type": "function",
				"function": map[string]interface{}{
					"name":      fc.Name,
					"arguments": string(argsJSON),
				},
			})
		}

		if part.FunctionResponse != nil {
			fr := part.FunctionResponse
			responseJSON, err := json.Marshal(fr.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}
			return map[string]interface{}{
				"role": "tool",
				"content": []map[string]interface{}{
					{
						"tool_call_id": fr.ID,
						"output":       string(responseJSON),
					},
				},
			}, nil
		}
	}

	if len(toolCalls) > 0 {
		return toolCalls, nil
	}

	if len(imageURLs) > 0 && len(texts) > 0 {
		content := make([]interface{}, 0, len(imageURLs)+len(texts))
		for _, text := range texts {
			content = append(content, map[string]interface{}{
				"type": "text",
				"text": text,
			})
		}
		for _, img := range imageURLs {
			content = append(content, img)
		}
		return content, nil
	}

	if len(imageURLs) > 0 {
		return imageURLs, nil
	}

	if len(texts) > 0 {
		return strings.Join(texts, ""), nil
	}

	return nil, nil
}

func (m *openAIModel) convertTools(tools map[string]any) ([]map[string]interface{}, error) {
	var result []map[string]interface{}
	for name, def := range tools {
		funcDecl, ok := def.(*genai.FunctionDeclaration)
		if !ok {
			continue
		}
		params, err := m.convertSchema(funcDecl.Parameters)
		if err != nil {
			return nil, err
		}
		result = append(result, map[string]interface{}{
			"type": "function",
			"name": name,
			"function": map[string]interface{}{
				"description": funcDecl.Description,
				"parameters":  params,
			},
		})
	}
	return result, nil
}

func (m *openAIModel) convertSchema(schema *genai.Schema) (interface{}, error) {
	if schema == nil {
		return nil, nil
	}
	return schema, nil
}

func (m *openAIModel) convertResponse(resp *chatCompletionsResponse) (*model.LLMResponse, error) {
	if len(resp.Choices) == 0 {
		return &model.LLMResponse{
			ErrorCode:    "NO_CHOICES",
			ErrorMessage: "No choices in response",
		}, nil
	}

	choice := resp.Choices[0]
	if choice.Message == nil {
		return &model.LLMResponse{
			ErrorCode:    "NO_MESSAGE",
			ErrorMessage: "No message in choice",
		}, nil
	}

	llmResp := &model.LLMResponse{
		Content:         m.convertResponseMessage(choice.Message),
		ThinkingContent: choice.Message.ReasoningContent,
		FinishReason:    genai.FinishReason(choice.FinishReason),
	}

	if resp.Usage != nil {
		llmResp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.Usage.PromptTokens),
			CandidatesTokenCount: int32(resp.Usage.CompletionTokens),
			TotalTokenCount:      int32(resp.Usage.TotalTokens),
		}
	}

	return llmResp, nil
}

func (m *openAIModel) convertResponseMessage(msg *responseMessage) *genai.Content {
	if msg == nil {
		return nil
	}

	var parts []*genai.Part
	if msg.Content != "" {
		parts = append(parts, &genai.Part{
			Text: msg.Content,
		})
	}

	for _, tc := range msg.ToolCalls {
		if tc.Function.Name != "" {
			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
					Args: m.parseArgs(tc.Function.Arguments),
				},
			})
		}
	}

	if len(parts) == 0 {
		parts = append(parts, &genai.Part{
			Text: "",
		})
	}

	return &genai.Content{
		Role:  "model",
		Parts: parts,
	}
}

func (m *openAIModel) parseArgs(arguments string) map[string]any {
	if arguments == "" {
		return nil
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return nil
	}
	return args
}

func (m *openAIModel) convertStreamChoice(c *choice, usage *usage) *model.LLMResponse {
	hasContent := c.Delta != nil && c.Delta.Content != ""
	hasReasoning := c.Delta != nil && c.Delta.ReasoningContent != ""
	isFinal := c.FinishReason != "" && c.FinishReason != "null"

	llmResp := &model.LLMResponse{
		Partial:         (hasContent || hasReasoning) && !isFinal,
		TurnComplete:    isFinal,
		FinishReason:    genai.FinishReason(c.FinishReason),
		ThinkingContent: getReasoningContent(c.Delta),
	}

	if c.Delta != nil && hasContent {
		llmResp.Content = m.convertResponseMessage(c.Delta)
	}

	if usage != nil {
		llmResp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(usage.PromptTokens),
			CandidatesTokenCount: int32(usage.CompletionTokens),
			TotalTokenCount:      int32(usage.TotalTokens),
		}
	}

	return llmResp
}

func getReasoningContent(delta *responseMessage) string {
	if delta == nil {
		return ""
	}
	return delta.ReasoningContent
}

func (m *openAIModel) parseStream(body io.Reader, yield func(*model.LLMResponse, error) bool) bool {
	reader := io.Reader(body)
	buf := make([]byte, 0, 4096)
	remainder := ""

	for {
		n, err := reader.Read(buf[:cap(buf)])
		if n > 0 {
			remainder += string(buf[:n])

			for {
				line, rest, ok := strings.Cut(remainder, "\n")
				if !ok {
					break
				}
				remainder = rest

				line = strings.TrimSpace(line)
				if !strings.HasPrefix(line, "data: ") {
					continue
				}

				if line == "data: [DONE]" {
					return true
				}

				dataStr := strings.TrimPrefix(line, "data: ")
				var chunk chatCompletionsResponse
				if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
					yield(nil, fmt.Errorf("failed to parse stream chunk: %w", err))
					return false
				}

				for _, c := range chunk.Choices {
					llmResp := m.convertStreamChoice(&c, chunk.Usage)

					if !yield(llmResp, nil) {
						return false
					}
				}
			}
		}

		if err != nil {
			if err == io.EOF && remainder != "" {
				remainder = strings.TrimSpace(remainder)
				if remainder == "data: [DONE]" {
					return true
				}
				if strings.HasPrefix(remainder, "data: ") {
					dataStr := strings.TrimPrefix(remainder, "data: ")
					var chunk chatCompletionsResponse
					if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
						yield(nil, fmt.Errorf("failed to parse final chunk: %w", err))
						return false
					}
					for _, c := range chunk.Choices {
						llmResp := m.convertStreamChoice(&c, chunk.Usage)
						if !yield(llmResp, nil) {
							return false
						}
					}
				}
			}
			return true
		}
	}
}
