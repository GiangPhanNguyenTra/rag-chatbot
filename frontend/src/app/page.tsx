"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function ChatbotPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Ref để cuộn lịch sử chat xuống cuối
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const API_URL =
    process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://127.0.0.1:8000";

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Hàm xử lý gửi tin nhắn
  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput("");
    setIsLoading(true);

    setMessages((prevMessages) => [
      ...prevMessages,
      { role: "assistant", content: "Đang trả lời..." },
    ]);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: userMessage.content }),
      });

      if (!response.ok) {
        throw new Error(`Lỗi HTTP! Status: ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
      };

      setMessages((prevMessages) => {
        const newMessages = [...prevMessages];
        newMessages.pop();
        return [...newMessages, assistantMessage];
      });
    } catch (error) {
      console.error("Lỗi khi gửi yêu cầu:", error);
      setMessages((prevMessages) => {
        const newMessages = [...prevMessages];
        newMessages.pop();
        return [
          ...newMessages,
          {
            role: "assistant",
            content: "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại.",
          },
        ];
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  useEffect(() => {
    setMessages([
      {
        role: "assistant",
        content: "Chào bạn! Tôi là Chatbot. Hãy hỏi tôi bất cứ điều gì.",
      },
    ]);
  }, []); // Chạy một lần khi component mount

  return (
    <div className="flex flex-col h-[80vh] max-w-xl mx-auto border border-gray-300 rounded-lg shadow-lg bg-white">
      <span className="text-xl font-bold text-center py-4  rounded-t-lg bg-purple-400 text-white ">
        AI Chatbot Tài Liệu
      </span>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`p-3 rounded-lg max-w-[80%] ${
              msg.role === "user"
                ? "bg-purple-100 text-right ml-auto"
                : "bg-gray-100 text-left mr-auto"
            }`}
          >
            {msg.content}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="p-4 border-t border-gray-200 flex items-center">
        <textarea
          id="userInput"
          className="flex-1 p-2 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:bg-gray-100"
          placeholder="Gõ câu hỏi của bạn..."
          rows={2} // Chiều cao ban đầu 2 dòng
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
        />
        <button
          id="sendButton"
          className="px-6 py-2 bg-purple-400 text-white rounded-r-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={sendMessage}
          disabled={isLoading}
        >
          {isLoading ? "Đang gửi..." : "Gửi"}
        </button>
      </div>
    </div>
  );
}
