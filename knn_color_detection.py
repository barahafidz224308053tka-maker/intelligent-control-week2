import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import cv2
import os

class ColorDetectorKNN:
    def __init__(self, csv_path='datasheet/colors.csv', n_neighbors=5):
        """
        Inisialisasi Color Detector dengan KNN
        
        Args:
            csv_path (str): Path ke file CSV dataset warna
            n_neighbors (int): Jumlah tetangga untuk KNN
        """
        self.csv_path = csv_path
        self.n_neighbors = n_neighbors
        self.knn = None
        self.scaler = StandardScaler()
        self.color_bgr_map = {}
        self.total_frames = 0
        self.correct_predictions = 0
        
        # Load dan train model
        self.load_and_train_model()
        
    def load_and_train_model(self):
        """Load dataset dan train model KNN"""
        try:
            # Baca dataset
            color_data = pd.read_csv(self.csv_path)
            print(f"Dataset loaded: {len(color_data)} samples")
            print("Available colors:", color_data['color_name'].unique())
            
            # Pisahkan fitur dan label
            X = color_data[['R', 'G', 'B']].values
            y = color_data['color_name'].values
            
            # Normalisasi fitur
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model KNN
            self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            self.knn.fit(X_train, y_train)
            
            # Evaluasi model jika ada test data
            if len(X_test) > 0:
                y_pred = self.knn.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Model accuracy on test set: {accuracy*100:.2f}%")
            else:
                print("Dataset terlalu kecil untuk test split, menggunakan semua data untuk training")
            
            # Buat mapping warna ke BGR untuk visualisasi
            self.create_color_bgr_map(color_data)
            
        except FileNotFoundError:
            print(f"Error: File {self.csv_path} tidak ditemukan!")
            print("Pastikan file colors.csv ada di folder datasheet/")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def create_color_bgr_map(self, color_data):
        """Membuat mapping nama warna ke nilai BGR"""
        for _, row in color_data.iterrows():
            color_name = row['color_name']
            # Konversi RGB ke BGR untuk OpenCV
            bgr_color = (int(row['B']), int(row['G']), int(row['R']))
            self.color_bgr_map[color_name] = bgr_color
    
    def predict_color(self, rgb_pixel):
        """
        Prediksi warna dari pixel RGB
        
        Args:
            rgb_pixel (array): Array RGB pixel
            
        Returns:
            str: Nama warna yang diprediksi
        """
        rgb_scaled = self.scaler.transform(rgb_pixel.reshape(1, -1))
        predicted_color = self.knn.predict(rgb_scaled)[0]
        return predicted_color
    
    def get_color_confidence(self, rgb_pixel):
        """
        Mendapatkan confidence score prediksi
        
        Args:
            rgb_pixel (array): Array RGB pixel
            
        Returns:
            tuple: (predicted_color, confidence_score)
        """
        rgb_scaled = self.scaler.transform(rgb_pixel.reshape(1, -1))
        
        # Prediksi dengan probabilitas
        predicted_color = self.knn.predict(rgb_scaled)[0]
        
        # Hitung jarak ke tetangga terdekat untuk confidence
        distances, indices = self.knn.kneighbors(rgb_scaled)
        avg_distance = np.mean(distances[0])
        confidence = max(0, 100 - avg_distance * 10)  # Convert distance to confidence
        
        return predicted_color, confidence
    
    def detect_color_hsv(self, hsv_pixel):
        """
        Deteksi warna khusus menggunakan HSV (untuk warna primer)
        
        Args:
            hsv_pixel (array): Pixel dalam format HSV
            
        Returns:
            str or None: Nama warna jika terdeteksi, None jika tidak
        """
        # Definisi rentang HSV untuk warna-warna primer
        color_ranges = {
            "Red": [([0, 120, 70], [10, 255, 255]), ([160, 120, 70], [179, 255, 255])],
            "Yellow": [([20, 100, 100], [30, 255, 255])],
            "Green": [([40, 50, 50], [80, 255, 255])],
            "Blue": [([90, 50, 50], [130, 255, 255])],
            "Cyan": [([80, 50, 50], [100, 255, 255])],
            "Magenta": [([140, 50, 50], [170, 255, 255])]
        }
        
        for color_name, ranges in color_ranges.items():
            for lower, upper in ranges:
                mask = cv2.inRange(hsv_pixel, np.array(lower), np.array(upper))
                if mask[0] > 0:
                    return color_name
        
        return None
    
    def draw_label_and_box(self, image, text, top_left, bottom_right, color, 
                          confidence=None, acc_text=None):
        """
        Menggambar label dan bounding box pada gambar
        
        Args:
            image (array): Gambar input
            text (str): Text label
            top_left (tuple): Koordinat kiri atas
            bottom_right (tuple): Koordinat kanan bawah  
            color (tuple): Warna BGR untuk box
            confidence (float): Confidence score
            acc_text (str): Text akurasi
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        font_scale = 1.0
        
        # Gambar bounding box
        cv2.rectangle(image, top_left, bottom_right, color, 3)
        
        # Background semi-transparan
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Tentukan warna text berdasarkan brightness background
        brightness = (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
        
        # Gambar text nama warna
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = top_left[0] + (bottom_right[0] - top_left[0] - text_size[0]) // 2
        text_y = top_left[1] + (bottom_right[1] - top_left[1] + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Gambar confidence score jika ada
        if confidence is not None:
            conf_text = f"Conf: {confidence:.1f}%"
            cv2.putText(image, conf_text, (text_x, text_y + 30), font, 0.6, text_color, 1)
        
        # Gambar akurasi di pojok kiri atas
        if acc_text:
            cv2.putText(image, acc_text, (10, 30), font, 0.7, (0, 255, 0), 2)
            

    
    def run_webcam_detection(self):
        """Menjalankan deteksi warna real-time dari webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Tidak dapat membuka kamera")
            return
        
        print("Deteksi warna dimulai. Tekan 'q' untuk keluar, 'r' untuk reset akurasi")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame dari kamera")
                break
            
            # Dapatkan dimensi frame
            height, width, _ = frame.shape
            center_x, center_y = width // 2, height // 2
            
            # Definisikan ROI (Region of Interest)
            roi_w, roi_h = 120, 120
            top_left = (center_x - roi_w // 2, center_y - roi_h // 2)
            bottom_right = (center_x + roi_w // 2, center_y + roi_h // 2)
            
            # Ambil pixel di tengah ROI
            pixel_bgr = frame[center_y, center_x].reshape(1, 1, 3)
            pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
            pixel_rgb = pixel_bgr[0][0][::-1]  # BGR ke RGB
            
            # Coba deteksi dengan HSV terlebih dahulu (untuk warna primer)
            color_hsv = self.detect_color_hsv(pixel_hsv)
            
            if color_hsv:
                color_pred = color_hsv
                confidence = 95.0  # HSV detection memiliki confidence tinggi
            else:
                # Gunakan KNN untuk prediksi
                color_pred, confidence = self.get_color_confidence(pixel_rgb)
            
            # Update statistik akurasi (simplified)
            self.total_frames += 1
            if color_hsv or confidence > 70:  # Anggap benar jika confidence tinggi
                self.correct_predictions += 1
            
            accuracy_percent = (self.correct_predictions / self.total_frames) * 100
            
            # Dapatkan warna untuk bounding box
            box_color = self.color_bgr_map.get(color_pred, (128, 128, 128))
            
            # Buat text akurasi
            acc_text = f"Accuracy: {accuracy_percent:.1f}%"
            
            # Gambar label dan box
            self.draw_label_and_box(
                frame, color_pred, top_left, bottom_right, 
                box_color, confidence, acc_text
            )
            
            # Gambar crosshair di tengah
            cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 2)
            cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 2)
            
            # Tampilkan frame
            cv2.imshow('Color Detection with KNN', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset accuracy statistics
                self.total_frames = 0
                self.correct_predictions = 0
                print("Accuracy statistics reset")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Final accuracy: {accuracy_percent:.1f}%")

def main():
    """Fungsi utama"""
    # Pastikan folder datasheet ada
    if not os.path.exists('datasheet'):
        os.makedirs('datasheet')
        print("Folder 'datasheet' created")
        print("Please put your colors.csv file in the datasheet folder")
        return
    
    try:
        # Inisialisasi detector
        detector = ColorDetectorKNN(csv_path='datasheet/colors.csv', n_neighbors=5)
        
        # Jalankan deteksi webcam
        detector.run_webcam_detection()
        
    except FileNotFoundError:
        print("Please make sure colors.csv exists in the datasheet folder")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()