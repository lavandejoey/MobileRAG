// -*- coding: utf-8 -*-
/**
 * @file: apps/chat_ui/src/App.jsx
 * @author: LIU Ziyi
 * @email: lavandejoey@outlook.com
 * @date: 2025/08/15
 * @version: 0.12.0
 */
import React, {useState, useEffect, useRef} from 'react';
import './App.css';

function App() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [sessionId, setSessionId] = useState(() => {
        const saved = localStorage.getItem('session_id');
        if (saved) return saved;
        const sid = crypto.randomUUID();
        localStorage.setItem('session_id', sid);
        return sid;
    });
    const [evidenceList, setEvidenceList] = useState([]);
    const [memoryList, setMemoryList] = useState([]);
    const [isGenerating, setIsGenerating] = useState(false); // New state for generation status
    const [backendStatus, setBackendStatus] = useState('checking'); // 'online', 'offline', 'checking'
    const messagesEndRef = useRef(null);
    const [reasoningAvailable, setReasoningAvailable] = useState(false);
    const [showReasoning, setShowReasoning] = useState(false);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({behavior: "smooth"});
    };

    // Effect to scroll to bottom on new messages
    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Effect to check backend status periodically
    useEffect(() => {
        const checkStatus = async () => {
            try {
                const response = await fetch('/api/status');
                if (response.ok) {
                    const data = await response.json();
                    setBackendStatus(data.status === 'online' ? 'online' : 'offline');
                } else {
                    setBackendStatus('offline');
                }
            } catch (error) {
                setBackendStatus('offline');
            }
        };

        checkStatus(); // Check immediately on mount
        const intervalId = setInterval(checkStatus, 5000); // Check every 5 seconds

        return () => clearInterval(intervalId); // Cleanup on unmount
    }, []);

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (input.trim() === '') return;

        const userMessage = {sender: 'user', text: input};
        setMessages((prevMessages) => [...prevMessages, userMessage]);
        setInput('');
        setEvidenceList([]); // Clear previous evidence
        setReasoningAvailable(false);
        setShowReasoning(false);
        setIsGenerating(true); // Set generating status to true

        let currentAssistantMessage = '';
        const updateAssistantMessage = (text) => {
            setMessages((prevMessages) => {
                const lastMessage = prevMessages[prevMessages.length - 1];
                if (lastMessage && lastMessage.sender === 'assistant') {
                    return [...prevMessages.slice(0, -1), {...lastMessage, text: text}];
                } else {
                    return [...prevMessages, {sender: 'assistant', text: text}];
                }
            });
        };

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({query: input, session_id: sessionId}),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const {done, value} = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, {stream: true});

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
                                } else if (data.reasoning_available) {
                                    setReasoningAvailable(true);
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

    const getStatusColor = () => {
        switch (backendStatus) {
            case 'online':
                return 'green';
            case 'offline':
                return 'red';
            case 'checking':
                return 'orange';
            default:
                return 'gray';
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>MobileRAG</h1>
                <div className="backend-status">
                    <span style={{color: getStatusColor()}}>●</span> {backendStatus}
                    <span style={{marginLeft: 12, opacity: 0.65, fontSize: 12}}>
                        session: {sessionId.slice(0, 8)}…
                    </span>
                </div>
            </header>
            <div className="chat-main">
                <div className="chat-container">
                    <div className="messages-display">
                        {messages.map((msg, index) => (
                            <div key={index} className={`message ${msg.sender}`}>
                                {msg.text}
                            </div>
                        ))}
                        {reasoningAvailable && (
                            <div className="message assistant" style={{background: '#f9fafb'}}>
                                <button onClick={() => setShowReasoning(s => !s)}>
                                    {showReasoning ? 'Hide reasoning (redacted)' : 'Show reasoning (redacted)'}
                                </button>
                                {showReasoning && (
                                    <div style={{marginTop: 8, fontStyle: 'italic', color: '#666'}}>
                                        Reasoning content is hidden by design. We don’t reveal chain-of-thought,
                                        but you can inspect cited evidence in the sidebar.
                                    </div>
                                )}
                            </div>
                        )}
                        {isGenerating && (
                            <div className="generating-spinner-container">
                                <div className="spinner"></div>
                            </div>
                        )}
                        <div ref={messagesEndRef}/>
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
