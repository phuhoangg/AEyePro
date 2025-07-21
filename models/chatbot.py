import asyncio
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid
import sqlite3
from models.recommend_agent import RecommendAgent
from models.retrieval_agent import RetrievalAgent
import os
from pathlib import Path

# Import Master Agent
from models.master_agent  import MasterAgent

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Loại tin nhắn"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Cấu trúc tin nhắn chat"""
    id: str
    message_type: MessageType
    content: str
    timestamp: datetime
    session_id: str
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Chuyển đổi thành dictionary"""
        return {
            "id": self.id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        """Tạo từ dictionary"""
        return cls(
            id=data["id"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=data["session_id"],
            metadata=data.get("metadata")
        )


@dataclass
class ChatSession:
    """Phiên chat"""
    id: str
    title: str
    created_at: datetime
    last_active: datetime
    message_count: int = 0
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Chuyển đổi thành dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "message_count": self.message_count,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatSession':
        """Tạo từ dictionary"""
        return cls(
            id=data["id"],
            title=data["title"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            message_count=data.get("message_count", 0),
            metadata=data.get("metadata")
        )


class ChatStorage:
    """Lưu trữ lịch sử chat"""

    def __init__(self, db_path: str = "chat_history.db"):
        """Khởi tạo storage"""
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Khởi tạo cơ sở dữ liệu"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Tạo bảng sessions
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_active TEXT NOT NULL,
                        message_count INTEGER DEFAULT 0,
                        metadata TEXT
                    )
                """)

                # Tạo bảng messages
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions (id)
                    )
                """)

                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def save_session(self, session: ChatSession) -> bool:
        """Lưu phiên chat"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions 
                    (id, title, created_at, last_active, message_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session.id,
                    session.title,
                    session.created_at.isoformat(),
                    session.last_active.isoformat(),
                    session.message_count,
                    json.dumps(session.metadata or {})
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False

    def save_message(self, message: ChatMessage) -> bool:
        """Lưu tin nhắn"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO messages 
                    (id, session_id, message_type, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    message.id,
                    message.session_id,
                    message.message_type.value,
                    message.content,
                    message.timestamp.isoformat(),
                    json.dumps(message.metadata or {})
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return False

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Lấy phiên chat"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
                row = cursor.fetchone()

                if row:
                    return ChatSession(
                        id=row[0],
                        title=row[1],
                        created_at=datetime.fromisoformat(row[2]),
                        last_active=datetime.fromisoformat(row[3]),
                        message_count=row[4],
                        metadata=json.loads(row[5]) if row[5] else None
                    )
                return None
        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return None

    def get_messages(self, session_id: str, limit: int = 100) -> List[ChatMessage]:
        """Lấy tin nhắn của phiên"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM messages 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (session_id, limit))

                messages = []
                for row in cursor.fetchall():
                    messages.append(ChatMessage(
                        id=row[0],
                        session_id=row[1],
                        message_type=MessageType(row[2]),
                        content=row[3],
                        timestamp=datetime.fromisoformat(row[4]),
                        metadata=json.loads(row[5]) if row[5] else None
                    ))

                return list(reversed(messages))  # Trả về theo thứ tự thời gian
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []

    def get_all_sessions(self, limit: int = 50) -> List[ChatSession]:
        """Lấy tất cả phiên chat"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM sessions 
                    ORDER BY last_active DESC 
                    LIMIT ?
                """, (limit,))

                sessions = []
                for row in cursor.fetchall():
                    sessions.append(ChatSession(
                        id=row[0],
                        title=row[1],
                        created_at=datetime.fromisoformat(row[2]),
                        last_active=datetime.fromisoformat(row[3]),
                        message_count=row[4],
                        metadata=json.loads(row[5]) if row[5] else None
                    ))

                return sessions
        except Exception as e:
            logger.error(f"Error getting all sessions: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """Xóa phiên chat"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False


class ChatBot:
    """ChatBot chính với khả năng ghi nhớ lịch sử và tích hợp Master Agent"""

    def __init__(self,
                 master_agent: MasterAgent,
                 storage_path: str = "chat_history.db",
                 max_history_length: int = 100):
        """
        Khởi tạo ChatBot

        Args:
            master_agent: Instance của MasterAgent
            storage_path: Đường dẫn file lưu trữ lịch sử
            max_history_length: Số lượng tin nhắn tối đa giữ trong memory
        """
        self.master_agent = master_agent
        recommend_engine = RecommendAgent()
        retrieval_agent = RetrievalAgent()

        self.master_agent.set_agents(recommend_engine, retrieval_agent)
        self.storage = ChatStorage(storage_path)
        self.max_history_length = max_history_length

        # Phiên chat hiện tại
        self.current_session: Optional[ChatSession] = None
        self.current_messages: List[ChatMessage] = []

        # Cấu hình
        self.system_message = "Bạn là trợ lý AI chuyên về sức khỏe và theo dõi công việc."

        logger.info("ChatBot initialized successfully")

    def create_new_session(self, title: Optional[str] = None) -> str:
        """Tạo phiên chat mới"""
        session_id = str(uuid.uuid4())
        now = datetime.now()

        if not title:
            title = f"Chat {now.strftime('%Y-%m-%d %H:%M')}"

        session = ChatSession(
            id=session_id,
            title=title,
            created_at=now,
            last_active=now
        )

        # Lưu phiên mới
        self.storage.save_session(session)

        # Đặt làm phiên hiện tại
        self.current_session = session
        self.current_messages = []

        # Reset Master Agent conversation
        self.master_agent.reset_conversation()

        logger.info(f"Created new session: {session_id}")
        return session_id

    def load_session(self, session_id: str) -> bool:
        """Tải phiên chat"""
        session = self.storage.get_session(session_id)
        if not session:
            logger.error(f"Session not found: {session_id}")
            return False

        # Tải tin nhắn
        messages = self.storage.get_messages(session_id, self.max_history_length)

        # Đặt làm phiên hiện tại
        self.current_session = session
        self.current_messages = messages

        # Reset Master Agent và load lịch sử
        self.master_agent.reset_conversation()
        self._load_history_to_master_agent()

        logger.info(f"Loaded session: {session_id}")
        return True

    def _load_history_to_master_agent(self):
        """Load lịch sử chat vào Master Agent"""
        try:
            from langchain.schema import HumanMessage, AIMessage

            for message in self.current_messages:
                if message.message_type == MessageType.USER:
                    self.master_agent.conversation_history.append(
                        HumanMessage(content=message.content)
                    )
                elif message.message_type == MessageType.ASSISTANT:
                    self.master_agent.conversation_history.append(
                        AIMessage(content=message.content)
                    )
        except Exception as e:
            logger.error(f"Error loading history to master agent: {e}")

    async def send_message(self, user_message: str, metadata: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Gửi tin nhắn và nhận phản hồi

        Args:
            user_message: Tin nhắn từ người dùng
            metadata: Metadata bổ sung

        Returns:
            Tuple[response, message_id]: Phản hồi và ID tin nhắn
        """
        # Tạo phiên mới nếu chưa có
        if not self.current_session:
            self.create_new_session()

        # Tạo tin nhắn user
        user_msg = ChatMessage(
            id=str(uuid.uuid4()),
            message_type=MessageType.USER,
            content=user_message,
            timestamp=datetime.now(),
            session_id=self.current_session.id,
            metadata=metadata
        )

        # Lưu tin nhắn user
        self.current_messages.append(user_msg)
        self.storage.save_message(user_msg)

        try:
            # Gửi đến Master Agent
            response = await self.master_agent.process_query(user_message)

            # Tạo tin nhắn phản hồi
            response_msg = ChatMessage(
                id=str(uuid.uuid4()),
                message_type=MessageType.ASSISTANT,
                content=response,
                timestamp=datetime.now(),
                session_id=self.current_session.id,
                metadata={"agent_status": self.master_agent.get_agent_status()}
            )

            # Lưu tin nhắn phản hồi
            self.current_messages.append(response_msg)
            self.storage.save_message(response_msg)

            # Cập nhật phiên
            self.current_session.last_active = datetime.now()
            self.current_session.message_count += 2
            self.storage.save_session(self.current_session)

            # Giữ lịch sử trong giới hạn
            self._trim_history()

            logger.info(f"Processed message successfully in session: {self.current_session.id}")
            return response, response_msg.id

        except Exception as e:
            logger.error(f"Error processing message: {e}")

            # Tạo tin nhắn lỗi
            error_msg = ChatMessage(
                id=str(uuid.uuid4()),
                message_type=MessageType.ASSISTANT,
                content=f"Xin lỗi, đã xảy ra lỗi khi xử lý tin nhắn: {str(e)}",
                timestamp=datetime.now(),
                session_id=self.current_session.id,
                metadata={"error": True}
            )

            self.current_messages.append(error_msg)
            self.storage.save_message(error_msg)

            return error_msg.content, error_msg.id

    def _trim_history(self):
        """Giữ lịch sử trong giới hạn"""
        if len(self.current_messages) > self.max_history_length:
            # Giữ lại các tin nhắn gần đây nhất
            self.current_messages = self.current_messages[-self.max_history_length:]

            # Cập nhật lịch sử Master Agent
            self.master_agent.reset_conversation()
            self._load_history_to_master_agent()

    def get_current_session(self) -> Optional[ChatSession]:
        """Lấy phiên hiện tại"""
        return self.current_session

    def get_current_messages(self) -> List[ChatMessage]:
        """Lấy tin nhắn phiên hiện tại"""
        return self.current_messages.copy()

    def get_all_sessions(self) -> List[ChatSession]:
        """Lấy tất cả phiên chat"""
        return self.storage.get_all_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Xóa phiên chat"""
        success = self.storage.delete_session(session_id)

        # Nếu xóa phiên hiện tại, tạo phiên mới
        if success and self.current_session and self.current_session.id == session_id:
            self.current_session = None
            self.current_messages = []
            self.master_agent.reset_conversation()

        return success

    def search_messages(self, query: str, session_id: Optional[str] = None) -> List[ChatMessage]:
        """Tìm kiếm tin nhắn"""
        # Đơn giản: tìm kiếm trong phiên hiện tại
        search_messages = self.current_messages if not session_id else self.storage.get_messages(session_id or "")

        results = []
        query_lower = query.lower()

        for message in search_messages:
            if query_lower in message.content.lower():
                results.append(message)

        return results

    def get_chat_statistics(self) -> Dict:
        """Lấy thống kê chat"""
        stats = {
            "total_sessions": len(self.get_all_sessions()),
            "current_session_messages": len(self.current_messages) if self.current_session else 0,
            "agent_status": self.master_agent.get_agent_status()
        }

        if self.current_session:
            stats["current_session_id"] = self.current_session.id
            stats["current_session_title"] = self.current_session.title

        return stats

    def update_session_title(self, session_id: str, new_title: str) -> bool:
        """Cập nhật tiêu đề phiên"""
        session = self.storage.get_session(session_id)
        if not session:
            return False

        session.title = new_title
        success = self.storage.save_session(session)

        # Cập nhật phiên hiện tại nếu cần
        if self.current_session and self.current_session.id == session_id:
            self.current_session.title = new_title

        return success

    def export_session(self, session_id: str, format: str = "json") -> Optional[str]:
        """Export phiên chat"""
        session = self.storage.get_session(session_id)
        if not session:
            return None

        messages = self.storage.get_messages(session_id)

        export_data = {
            "session": session.to_dict(),
            "messages": [msg.to_dict() for msg in messages]
        }

        if format == "json":
            return json.dumps(export_data, ensure_ascii=False, indent=2)

        # Có thể thêm format khác (txt, csv, etc.)
        return None

    def get_agent_status(self) -> Dict:
        """Lấy trạng thái agent"""
        return self.master_agent.get_agent_status()

    async def close(self):
        """Đóng chatbot"""
        # Có thể thêm cleanup logic ở đây
        logger.info("ChatBot closed")


