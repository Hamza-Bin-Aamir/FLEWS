import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './Chatbot.scss';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  recommendations?: string[];
}

interface ChatbotProps {
  floodData?: any;
  currentLocation?: [number, number];
}

const Chatbot: React.FC<ChatbotProps> = ({ floodData, currentLocation }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm your FLEWS assistant. I can help you with flood status updates, safety recommendations, and answer your questions about flood risks in Pakistan.",
      sender: 'bot',
      timestamp: new Date(),
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    try {
      // Send message to backend
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputText,
          context: {
            location: currentLocation,
            flood_data: floodData,
          }
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();

      // Simulate typing delay for better UX
      setTimeout(() => {
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: data.response,
          sender: 'bot',
          timestamp: new Date(),
          recommendations: data.recommendations,
        };

        setMessages(prev => [...prev, botMessage]);
        setIsTyping(false);
      }, 500);
    } catch (error) {
      console.error('Chat error:', error);
      
      // Fallback to local response if backend is unavailable
      setTimeout(() => {
        const fallbackMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: generateLocalResponse(inputText),
          sender: 'bot',
          timestamp: new Date(),
        };

        setMessages(prev => [...prev, fallbackMessage]);
        setIsTyping(false);
      }, 500);
    }
  };

  // Fallback local response generator
  const generateLocalResponse = (query: string): string => {
    const lowerQuery = query.toLowerCase();

    if (lowerQuery.includes('status') || lowerQuery.includes('safe')) {
      return "Based on current data, I can help you check flood status. Please make sure you're connected to the backend server for real-time updates.";
    } else if (lowerQuery.includes('help') || lowerQuery.includes('what can you')) {
      return "I can help you with:\n• Current flood status in your area\n• Safety recommendations\n• Evacuation guidance\n• Real-time alerts\n\nJust ask me anything about flood safety!";
    } else if (lowerQuery.includes('emergency') || lowerQuery.includes('evacuate')) {
      return "🚨 If you're in immediate danger:\n• Move to higher ground immediately\n• Call emergency services: 1122\n• Follow official evacuation orders\n• Take essential items only\n• Stay informed via radio/official channels";
    } else {
      return "I'm here to help with flood-related information. You can ask about flood status, safety tips, or emergency procedures.";
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const quickActions = [
    { label: "Check my area status", query: "What's the flood status in my area?" },
    { label: "Safety tips", query: "Give me flood safety tips" },
    { label: "Emergency help", query: "What should I do in an emergency?" },
  ];

  return (
    <>
      {/* Floating Chat Button */}
      <motion.button
        className="chat-button"
        onClick={() => setIsOpen(!isOpen)}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        aria-label="Toggle chat"
      >
        {isOpen ? '✕' : '💬'}
      </motion.button>

      {/* Chat Window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="chatbot-container"
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            {/* Header */}
            <div className="chatbot-header">
              <div className="header-info">
                <span className="bot-avatar">🤖</span>
                <div>
                  <h3>FLEWS Assistant</h3>
                  <span className="status-indicator">● Online</span>
                </div>
              </div>
              <button
                className="minimize-button"
                onClick={() => setIsOpen(false)}
                aria-label="Minimize chat"
              >
                −
              </button>
            </div>

            {/* Messages */}
            <div className="chatbot-messages">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`message ${message.sender === 'user' ? 'message-user' : 'message-bot'}`}
                >
                  <div className="message-content">
                    <p>{message.text}</p>
                    {message.recommendations && message.recommendations.length > 0 && (
                      <div className="recommendations">
                        <strong>📋 Recommendations:</strong>
                        <ul>
                          {message.recommendations.map((rec, idx) => (
                            <li key={idx}>{rec}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                  <span className="message-time">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
              ))}

              {isTyping && (
                <div className="message message-bot typing">
                  <div className="message-content">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Quick Actions */}
            {messages.length <= 2 && (
              <div className="quick-actions">
                {quickActions.map((action, idx) => (
                  <button
                    key={idx}
                    className="quick-action-button"
                    onClick={() => {
                      setInputText(action.query);
                      setTimeout(() => handleSendMessage(), 100);
                    }}
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            )}

            {/* Input */}
            <div className="chatbot-input">
              <input
                ref={inputRef}
                type="text"
                placeholder="Ask me anything about flood safety..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isTyping}
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputText.trim() || isTyping}
                aria-label="Send message"
              >
                ➤
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default Chatbot;
