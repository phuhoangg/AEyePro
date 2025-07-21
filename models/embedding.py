import pandas as pd
import numpy as np
import faiss
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import os


class HealthDataEmbedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Khởi tạo class với mô hình embedding nhẹ

        Args:
            model_name: Tên mô hình SentenceTransformer (mặc định là all-MiniLM-L6-v2 - 80MB)
        """
        self.model = SentenceTransformer(model_name)
        self.scaler = StandardScaler()
        self.feature_columns = [
            'blink_per_minute', 'avg_dist', 'avg_head_tilt', 'avg_head_side',
            'avg_shoulder_angle', 'avg_screen_brightness', 'avg_ambient_brightness',
            'number_of_drowsiness', 'bad_posture_count', 'session_duration_seconds'
        ]
        self.index = None
        self.metadata = []

    def _abs_path(self, path):
        if os.path.isabs(path):
            return path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, path)

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load dữ liệu từ CSV file

        Args:
            csv_path: Đường dẫn tới file CSV

        Returns:
            DataFrame chứa dữ liệu
        """
        csv_path = self._abs_path(csv_path)
        try:
            df = pd.read_csv(csv_path)
            # Tự động đổi tên cột avg_distance thành avg_dist nếu có
            if 'avg_distance' in df.columns and 'avg_dist' not in df.columns:
                df = df.rename(columns={'avg_distance': 'avg_dist'})
            print(f"Đã load {len(df)} records từ {csv_path}")
            return df
        except Exception as e:
            print(f"Lỗi khi load dữ liệu: {e}")
            return pd.DataFrame()

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[Dict]]:
        """
        Tiền xử lý dữ liệu và tạo metadata

        Args:
            df: DataFrame chứa dữ liệu gốc

        Returns:
            Tuple of (normalized_features, metadata_list)
        """
        # Chuẩn hóa dữ liệu số
        numeric_features = df[self.feature_columns].values
        normalized_features = self.scaler.fit_transform(numeric_features)

        # Tạo metadata cho mỗi session
        metadata_list = []
        for idx, row in df.iterrows():
            metadata = {
                'session_id': row['session_id'],
                'begin_timestamp': row['begin_timestamp'],
                'end_timestamp': row['end_timestamp'],
                'session_duration_minutes': row['session_duration_seconds'] / 60.0,
                'health_metrics': {
                    'blink_per_minute': float(row['blink_per_minute']),
                    'avg_distance': float(row['avg_dist']),
                    'avg_head_tilt': float(row['avg_head_tilt']),
                    'avg_head_side': float(row['avg_head_side']),
                    'avg_shoulder_angle': float(row['avg_shoulder_angle']),
                    'screen_brightness': float(row['avg_screen_brightness']),
                    'ambient_brightness': float(row['avg_ambient_brightness']),
                    'drowsiness_count': int(row['number_of_drowsiness']),
                    'bad_posture_count': int(row['bad_posture_count'])
                },
                'derived_metrics': {
                    'posture_quality_score': self._calculate_posture_score(row),
                    'attention_score': self._calculate_attention_score(row),
                    'environment_score': self._calculate_environment_score(row)
                },
                'time_features': {
                    'hour_of_day': pd.to_datetime(row['begin_timestamp']).hour,
                    'day_of_week': pd.to_datetime(row['begin_timestamp']).dayofweek,
                    'is_weekend': pd.to_datetime(row['begin_timestamp']).dayofweek >= 5
                }
            }
            metadata_list.append(metadata)

        return normalized_features, metadata_list

    def _calculate_posture_score(self, row) -> float:
        """Tính điểm tư thế (0-100, cao = tốt)"""
        # Normalize các chỉ số tư thế
        head_tilt_score = max(0, 100 - abs(row['avg_head_tilt']) * 2)
        head_side_score = max(0, 100 - abs(row['avg_head_side']) * 2)
        shoulder_score = max(0, 100 - abs(row['avg_shoulder_angle']) * 2)
        bad_posture_penalty = min(row['bad_posture_count'] * 2, 50)

        return max(0, (head_tilt_score + head_side_score + shoulder_score) / 3 - bad_posture_penalty)

    def _calculate_attention_score(self, row) -> float:
        """Tính điểm tập trung (0-100, cao = tốt)"""
        # Blink rate lý tưởng khoảng 15-20 lần/phút
        blink_score = max(0, 100 - abs(row['blink_per_minute'] - 17.5) * 3)
        drowsiness_penalty = row['number_of_drowsiness'] * 20

        return max(0, blink_score - drowsiness_penalty)

    def _calculate_environment_score(self, row) -> float:
        """Tính điểm môi trường (0-100, cao = tốt)"""
        # Tỷ lệ brightness lý tưởng
        brightness_ratio = row['avg_screen_brightness'] / max(row['avg_ambient_brightness'], 1)
        brightness_score = max(0, 100 - abs(brightness_ratio - 0.5) * 100)

        return brightness_score

    def create_text_descriptions(self, metadata_list: List[Dict]) -> List[str]:
        """
        Tạo mô tả văn bản cho mỗi session để embedding

        Args:
            metadata_list: List các metadata dict

        Returns:
            List các mô tả văn bản
        """
        descriptions = []
        for meta in metadata_list:
            health = meta['health_metrics']
            derived = meta['derived_metrics']
            time_info = meta['time_features']

            # Đánh giá mức độ
            posture_level = "tốt" if derived['posture_quality_score'] > 70 else "trung bình" if derived[
                                                                                                    'posture_quality_score'] > 40 else "kém"
            attention_level = "tốt" if derived['attention_score'] > 70 else "trung bình" if derived[
                                                                                                'attention_score'] > 40 else "kém"

            # Thời gian trong ngày
            time_period = "sáng" if time_info['hour_of_day'] < 12 else "chiều" if time_info[
                                                                                      'hour_of_day'] < 18 else "tối"

            description = f"""
            Phiên làm việc {meta['session_id'][:8]}.
            Tư thế làm việc: {posture_level} (nghiêng đầu {health['avg_head_tilt']:.1f}°, vai {health['avg_shoulder_angle']:.1f}°, 
            {health['bad_posture_count']} lần tư thế xấu).
            Mức độ tập trung: {attention_level} (nhấp nháy {health['blink_per_minute']:.1f} lần/phút, 
            {health['drowsiness_count']} lần buồn ngủ).
            Khoảng cách màn hình: {health['avg_distance']:.1f}cm.
            Ánh sáng: màn hình {health['screen_brightness']:.0f}, môi trường {health['ambient_brightness']:.0f}.
            """.strip()

            descriptions.append(description)

        return descriptions

    def create_embeddings(self, df: pd.DataFrame) -> None:
        # Tiền xử lý và tạo text_embeddings…
        normalized_features, metadata_list = self.preprocess_data(df)
        text_descriptions = self.create_text_descriptions(metadata_list)
        text_embeddings = self.model.encode(text_descriptions)

        # Kết hợp và ép kiểu về float32 ngay sau khi hstack
        combined = np.hstack([normalized_features, text_embeddings])
        combined = combined.astype('float32', copy=False)  # ① ép kiểu float32
        combined = np.ascontiguousarray(combined)  # ② đảm bảo C‑contiguous

        # Khởi tạo FAISS index với phép đo inner-product
        dimension = combined.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Chuẩn hoá L2 toàn bộ embeddings
        # (bây giờ chắc chắn là float32 & C‑contiguous, nên sẽ không lỗi)
        faiss.normalize_L2(combined)

        # Thêm embeddings vào index
        self.index.add(combined)

        # Lưu metadata
        self.metadata = metadata_list

        print(f"Đã tạo {combined.shape[0]} embeddings với {dimension} dims")

    def search_similar_sessions(self, query_text: str, k: int = 5) -> List[Dict]:
        """
        Tìm kiếm các session tương tự dựa trên text query
        """
        if self.index is None:
            print("Chưa tạo embeddings. Vui lòng chạy create_embeddings() trước.")
            return []

        # Tạo embedding cho query
        query_embedding = self.model.encode([query_text])

        # Pad với zeros để match dimension
        numeric_dim = len(self.feature_columns)
        zero_padding = np.zeros((1, numeric_dim))
        query_vector = np.hstack([zero_padding, query_embedding])

        # SỬA LỖI: Ép kiểu về float32 và đảm bảo C-contiguous NGAY LẬP TỨC
        query_vector = query_vector.astype('float32')
        query_vector = np.ascontiguousarray(query_vector)

        # Normalize (Bây giờ sẽ không còn lỗi)
        faiss.normalize_L2(query_vector)

        # Search (Không cần ép kiểu ở đây nữa)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                result = {
                    'rank': i + 1,
                    'similarity_score': float(score),
                    'metadata': self.metadata[idx]
                }
                results.append(result)

        return results

    def search_by_metrics(self,
                          posture_score_min: float = None,
                          attention_score_min: float = None,
                          duration_min: float = None,
                          hour_range: Tuple[int, int] = None) -> List[Dict]:
        """
        Tìm kiếm sessions dựa trên các chỉ số cụ thể

        Args:
            posture_score_min: Điểm tư thế tối thiểu
            attention_score_min: Điểm tập trung tối thiểu
            duration_min: Thời lượng tối thiểu (phút)
            hour_range: Khoảng thời gian trong ngày (start_hour, end_hour)

        Returns:
            List các session thỏa mãn điều kiện
        """
        results = []

        for meta in self.metadata:
            # Kiểm tra điều kiện
            if posture_score_min and meta['derived_metrics']['posture_quality_score'] < posture_score_min:
                continue
            if attention_score_min and meta['derived_metrics']['attention_score'] < attention_score_min:
                continue
            if duration_min and meta['session_duration_minutes'] < duration_min:
                continue
            if hour_range:
                hour = meta['time_features']['hour_of_day']
                if not (hour_range[0] <= hour <= hour_range[1]):
                    continue

            results.append(meta)

        return results

    def save_index(self, index_path: str = "data/health_data_index.faiss",
                   metadata_path: str = "data/health_data_metadata.pkl",
                   scaler_path: str = "data/health_data_scaler.pkl") -> None:
        """
        Lưu FAISS index và metadata

        Args:
            index_path: Đường dẫn lưu FAISS index
            metadata_path: Đường dẫn lưu metadata
            scaler_path: Đường dẫn lưu scaler
        """
        index_path = self._abs_path(index_path)
        metadata_path = self._abs_path(metadata_path)
        scaler_path = self._abs_path(scaler_path)

        if self.index is not None:
            faiss.write_index(self.index, index_path)
            print(f"Đã lưu FAISS index tại {index_path}")

        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Đã lưu metadata tại {metadata_path}")

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Đã lưu scaler tại {scaler_path}")

    def load_index(self, index_path: str = "data/health_data_index.faiss",
                   metadata_path: str = "data/health_data_metadata.pkl",
                   scaler_path: str = "data/health_data_scaler.pkl") -> None:
        """
        Load FAISS index và metadata đã lưu

        Args:
            index_path: Đường dẫn FAISS index
            metadata_path: Đường dẫn metadata
            scaler_path: Đường dẫn scaler
        """
        index_path = self._abs_path(index_path)
        metadata_path = self._abs_path(metadata_path)
        scaler_path = self._abs_path(scaler_path)

        try:
            self.index = faiss.read_index(index_path)
            print(f"Đã load FAISS index từ {index_path}")

            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Đã load metadata từ {metadata_path}")

            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Đã load scaler từ {scaler_path}")

        except Exception as e:
            print(f"Lỗi khi load: {e}")

    def get_statistics(self) -> Dict:
        """
        Thống kê tổng quan về dữ liệu

        Returns:
            Dict chứa các thống kê
        """
        if not self.metadata:
            return {}

        posture_scores = [m['derived_metrics']['posture_quality_score'] for m in self.metadata]
        attention_scores = [m['derived_metrics']['attention_score'] for m in self.metadata]
        durations = [m['session_duration_minutes'] for m in self.metadata]

        stats = {
            'total_sessions': len(self.metadata),
            'average_posture_score': np.mean(posture_scores),
            'average_attention_score': np.mean(attention_scores),
            'average_duration_minutes': np.mean(durations),
            'posture_score_distribution': {
                'good': sum(1 for s in posture_scores if s > 70),
                'average': sum(1 for s in posture_scores if 40 <= s <= 70),
                'poor': sum(1 for s in posture_scores if s < 40)
            },
            'attention_score_distribution': {
                'good': sum(1 for s in attention_scores if s > 70),
                'average': sum(1 for s in attention_scores if 40 <= s <= 70),
                'poor': sum(1 for s in attention_scores if s < 40)
            }
        }

        return stats


# def main():
#     """
#     Hàm chính để demo sử dụng
#     """
#     # Khởi tạo
#     embedder = HealthDataEmbedding()
#
#     # Load dữ liệu
#     df = embedder.load_data("data/summary.csv")
#
#     if df.empty:
#         print("Không thể load dữ liệu. Vui lòng kiểm tra đường dẫn file.")
#         return
#
#     # Tạo embeddings
#     embedder.create_embeddings(df)
#
#     # Lưu index
#     embedder.save_index()
#
#     # Thống kê
#     stats = embedder.get_statistics()
#     print("\n=== THỐNG KÊ TỔNG QUAN ===")
#     print(f"Tổng số sessions: {stats['total_sessions']}")
#     print(f"Điểm tư thế trung bình: {stats['average_posture_score']:.1f}")
#     print(f"Điểm tập trung trung bình: {stats['average_attention_score']:.1f}")
#     print(f"Thời lượng trung bình: {stats['average_duration_minutes']:.1f} phút")
#
#     # Demo tìm kiếm
#     print("\n=== DEMO TÌM KIẾM ===")
#
#     # Tìm kiếm bằng text
#     results = embedder.search_similar_sessions("phiên làm việc có tư thế xấu", k=3)
#     print("\nTìm kiếm 'phiên làm việc có tư thế xấu':")
#     for result in results:
#         meta = result['metadata']
#         print(f"- Session {meta['session_id'][:8]}: Score {result['similarity_score']:.3f}")
#         print(f"  Tư thế: {meta['derived_metrics']['posture_quality_score']:.1f}, "
#               f"Tập trung: {meta['derived_metrics']['attention_score']:.1f}")
#
#     # Tìm kiếm bằng metrics
#     good_sessions = embedder.search_by_metrics(posture_score_min=60, attention_score_min=60)
#     print(f"\nSessions có tư thế và tập trung tốt: {len(good_sessions)}")
#
#     print("\n=== HOÀN THÀNH ===")
#     print("Embeddings đã được tạo và lưu thành công!")
#     print("Bạn có thể sử dụng các file đã lưu để tích hợp với agent.")


# if __name__ == "__main__":
#     main()