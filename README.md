# AEyePro

## Giới thiệu

**AEyePro** là ứng dụng hỗ trợ theo dõi sức khỏe mắt, tư thế và trạng thái làm việc của người dùng máy tính, tích hợp AI để tư vấn, cảnh báo và tổng hợp dữ liệu sức khỏe cá nhân. Ứng dụng sử dụng camera để nhận diện chớp mắt, phát hiện buồn ngủ, phân tích tư thế, đồng thời cung cấp dashboard thống kê và trợ lý AI trả lời các câu hỏi liên quan đến sức khỏe/làm việc.

## Tính năng nổi bật

- **Giám sát thời gian thực**: Theo dõi chớp mắt, phát hiện buồn ngủ, phân tích tư thế ngồi, đo khoảng cách mắt - màn hình, cảnh báo khi phát hiện dấu hiệu nguy hiểm (buồn ngủ, tư thế xấu, ánh sáng yếu...).
- **Thu thập & tổng hợp dữ liệu**: Lưu lại dữ liệu sức khỏe từng phiên làm việc, xuất báo cáo, dashboard trực quan (biểu đồ chớp mắt, buồn ngủ, tư thế, thời gian làm việc...).
- **Trợ lý AI**: Tích hợp AI (Llama, Langchain) trả lời câu hỏi về sức khỏe, tư thế, lịch sử làm việc, đưa ra khuyến nghị cá nhân hóa.
- **Cảnh báo thông minh**: Nhắc nhở nghỉ ngơi, thay đổi tư thế, điều chỉnh ánh sáng, phòng tránh mỏi mắt và các vấn đề sức khỏe liên quan.
- **Giao diện hiện đại**: Xây dựng bằng CustomTkinter, trực quan, dễ sử dụng, hỗ trợ chuyển đổi giữa các module (Giám sát, Tổng kết, AI Assistant).

## Cấu trúc
![pipeline](https://github.com/user-attachments/assets/4d3855c8-6d77-41fd-b229-dfddca515ff0)

## Yêu cầu hệ thống

- Windows 10/11 (khuyến nghị)
- Python 3.12
- Camera máy tính (webcam)
- Khuyến nghị: GPU để tăng tốc AI (không bắt buộc, project hiện đang được cài đặt để chạy trên CPU)

## Hướng dẫn cài đặt

1. **Cài đặt Anaconda/Miniconda** (nếu chưa có).
2. Mở terminal/cmd tại thư mục dự án, tạo môi trường và cài đặt phụ thuộc:
   ```bash
   conda env create -f environment.yml
   conda activate AEyePro
   ```
3. **Tải file mô hình .gguf thủ công:**
   - Do file mô hình AI (.gguf) có dung lượng lớn (~3GB) và Git LFS không cho phép tải lên kho mã nguồn, bạn cần tải file này thủ công.
   - Tìm và tải file `Llama-3.2-3B-Instruct-Q8_0.gguf` từ nguồn uy tín (ví dụ: HuggingFace, repo chính thức của Llama hoặc do nhóm phát triển cung cấp).
   - Sau khi tải xong, đặt file vào thư mục `models/` của dự án:
     ```
     AEyeApp/models/Llama-3.2-3B-Instruct-Q8_0.gguf
     ```
   - Nếu không có file này, tính năng AI Assistant sẽ không hoạt động.
   - Có thể thay đổi mô hình khác nếu thiết bị cho phép và thay đổi đường dẫn trong các file ở thư mục `models`

4. Đảm bảo đã cắm webcam và cho phép truy cập camera.

## Hướng dẫn sử dụng

1. **Chạy ứng dụng:**
   ```bash
   python main.py
   ```
2. **Các module chính:**
   - **Monitoring**: Theo dõi sức khỏe mắt, tư thế, trạng thái buồn ngủ theo thời gian thực. Nhận cảnh báo trực tiếp trên giao diện.
   - **Summary**: Xem lại lịch sử, biểu đồ tổng hợp các chỉ số sức khỏe, lọc theo ngày/tháng.
   - **AI Assistant**: Đặt câu hỏi cho trợ lý AI về sức khỏe, tư thế, lịch sử làm việc, nhận khuyến nghị cá nhân hóa.

3. **Cấu hình**: Thay đổi các ngưỡng, thông số nhận diện trong file `config/settings.json` nếu cần (không khuyến khích với người dùng phổ thông).

## Cấu trúc thư mục

```
AEyePro/
│
├── main.py                  # Điểm khởi động ứng dụng
├── environment.yml          # File khai báo môi trường & phụ thuộc
├── config/
│   └── settings.json        # File cấu hình các ngưỡng nhận diện
├── core/                    # Các module lõi: nhận diện mắt, chớp mắt, buồn ngủ, tư thế, đánh giá sức khỏe
├── GUI/                     # Giao diện người dùng (CustomTkinter)
├── Manager/                 # Quản lý cảnh báo, biểu đồ
├── models/                  # Các agent AI, mô hình nhúng, file mô hình Llama
├── service/                 # Cấu hình bảo mật, dịch vụ phụ trợ
├── utils/                   # Tiện ích: xử lý dữ liệu, cấu hình, hiệu chuẩn
├── data/                    # Lưu trữ dữ liệu realtime và tổng hợp
└── Execution/               # Quản lý luồng thực thi song song
```

## Lưu ý

- Ứng dụng KHÔNG gửi dữ liệu cá nhân ra ngoài, mọi dữ liệu được lưu cục bộ.
- Để AI hoạt động tốt, cần đủ tài nguyên RAM (~8GB+) và ổ cứng trống (file mô hình ~3GB).
- Nếu gặp lỗi camera, hãy kiểm tra quyền truy cập hoặc thử đổi `camera_index` trong `config/settings.json`.

## Đóng góp & phát triển

- Để báo lỗi hoặc đề xuất cải tiến, tạo [issue](https://github.com/phuhoangg/AEyePro/issues) trên GitHub
- Để đóng góp code, submit [pull request](https://github.com/phuhoangg/AEyePro/pulls) với detailed description.

## License

Dự án được cấp phép theo [MIT License](https://opensource.org/licenses/MIT). Xem chi tiết trong file [LICENSE](LICENSE).

---
