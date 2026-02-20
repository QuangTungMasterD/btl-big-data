## 1. 📖 Giới thiệu.
<p>
    Dự án này thực hiện phân tích dữ liệu nhân sự dựa trên HR Analytics Dataset (Kaggle) nhằm nghiên cứu các yếu tố ảnh hưởng đến quyết định nghỉ việc của nhân viên và xây dựng mô hình dự đoán khả năng nghỉ việc (Attrition)
</p>

## 2. 🕷️ Công nghệ sử dụng.
<div align="center">

[![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
</div>

## 3. ⚙️ Cài đặt.
### 3.1. Cài đặt công cụ, môi trường và các thư viện cần thiết
#### 3.1.1. Tải project.

```
git clone https://github.com/QuangTungMasterD/btl-big-data.git
```

#### 3.1.2. Môi trường ảo.

- Cài đặt và khởi động môi trường máy ảo ubuntu.
- Cài đặt python/pip trên môi trường ubuntu.

```
sudo apt install python3-pip
```

- Khởi tạo môi trường ảo

```
python3.10 -m venv .venv
```

- Thay đổi trình thông dịch sang môi trường ảo

```
source .venv/bin/activate
```

- Chạy requirements.txt để cài đặt tiếp các thư viện được yêu cầu

```
pip3 install -r requirements.txt
```

#### 3.1.3. Tạo thư mục.

#### Tạo thư mục outputs.
- Trong outputs tạo thư mục **figures**.
- Trong outputs tạo thư mục **models**.
- Trong outputs tạo thư mục **tables**.

#### Tạo thư mục data.
- Trong outputs tạo thư mục **processed**.
- Trong outputs tạo thư mục **raw**.

#### 3.1.4. Tải dữ liệu.

<h3>
    <p>
        Tải file dữ liệu tại<a href="https://www.kaggle.com/datasets/anshika2301/hr-analytics-dataset">
            hr analytics dataset
        </a>
    </P>
</h3>

- File dữ liệu sẽ được lưu tại data/raw.

## 3.2. Chạy chương trình.
### 3.2.1. Chạy pipeline.

```
python3 -m scripts.run_pipeline
```

### 3.2.2. Chạy papermill.

```
python3 -m scripts.run_papermill
```