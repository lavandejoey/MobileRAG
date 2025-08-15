// -*- coding: utf-8 -*-
/**
 * @author: LIU Ziyi
 * @email: lavandejoey@outlook.com
 * @date: 2025/08/15
 * @version: 0.12.0
 */
import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState('test_session'); // Default session ID
  const [evidenceList, setEvidenceList] = useState([]);
  const [memoryList, setMemoryList] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false); // New state for generation status
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (input.trim() === '') return;

    const userMessage = { sender: 'user', text: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput('');
    setEvidenceList([]); // Clear previous evidence
    setIsGenerating(true); // Set generating status to true

    let currentAssistantMessage = '';
    const updateAssistantMessage = (text) => {
      setMessages((prevMessages) => {
        const lastMessage = prevMessages[prevMessages.length - 1];
        if (lastMessage && lastMessage.sender === 'assistant') {
          return [...prevMessages.slice(0, -1), { ...lastMessage, text: text }];
        } else {
          return [...prevMessages, { sender: 'assistant', text: text }];
        }
      });
    };

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: input, session_id: sessionId }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });

        // Process each line as a separate SSE event
        chunk.split('\n').forEach(line => {
          if (line.startsWith('data:')) {
            const jsonString = line.substring(5).trim();
            if (jsonString) {
              try {
                const data = JSON.parse(jsonString);
                if (data.answer) {
                  currentAssistantMessage += data.answer;
                  updateAssistantMessage(currentAssistantMessage);
                } else if (data.evidence) {
                  setEvidenceList(prev => [...prev, data.evidence]);
                } else if (data.memory) {
                  setMemoryList(prev => [...prev, data.memory]);
                }
              } catch (error) {
                console.error("Error parsing SSE JSON:", error, "JSON string:", jsonString);
              }
            }
          }
        });
      }
    } catch (error) {
      console.error('Error sending message:', error);
      updateAssistantMessage('Error: Could not connect to the server.');
    } finally {
      setIsGenerating(false); // Set generating status to false when done or error
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>MobileRAG Chat</h1>
      </header>
      <div className="chat-main">
        <div className="chat-container">
          <div className="messages-display">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                {msg.text}
              </div>
            ))}
            {isGenerating && (
              <div className="message assistant generating">
                Thinking...
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <form onSubmit={handleSendMessage} className="message-input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              disabled={isGenerating}
            />
            <button type="submit" disabled={!input.trim() || isGenerating}>Send</button>
          </form>
        </div>
        <div className="sidebar">
          <h2>Evidence</h2>
          {evidenceList.length === 0 ? (
            <p>No evidence found for this query.</p>
          ) : (
            <ul>
              {evidenceList.map((ev, index) => (
                <li key={index}>{JSON.stringify(ev)}</li> // Display raw JSON for now
              ))}
            </ul>
          )}
          <h2>Memory</h2>
          {memoryList.length === 0 ? (
            <p>No memory items found.</p>
          ) : (
            <ul>
              {memoryList.map((mem, index) => (
                <li key={index}>{JSON.stringify(mem)}</li> // Display raw JSON for now
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
