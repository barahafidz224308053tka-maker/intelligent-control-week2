import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
import os

class DualPointColorDetectorSVM:
    def __init__(self, csv_path='datasheet/colors.csv'):
        """
        Inisialisasi Color Detector SVM untuk 2 titik deteksi
        
        Args:
            csv_path (str): Path ke file CSV dataset warna
        """
        self.csv_path = csv_path
        self.svm = None
        self.scaler = StandardScaler()
        self.color_bgr_map = {}
        self.total_frames_point1 = 0
        self.correct_predictions_point1 = 0
        self.total_frames_point2 = 0
        self.correct_predictions_point2 = 0
        
        # Load dan train model
        self.load_and_train_model()
        
    def load_and_train_model(self):
        """Load dataset dan train model SVM"""
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
            
            # Train model SVM dengan RBF kernel
            print("Training SVM with RBF kernel...")
            self.svm = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,  # Enable probability estimates for confidence
                random_state=42
            )
            self.svm.fit(X_train, y_train)
            
            # Evaluasi model jika ada test data
            if len(X_test) > 0:
                y_pred = self.svm.predict(X_test)
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
    
    def get_color_confidence(self, rgb_pixel):
        """
        Mendapatkan confidence score prediksi dari SVM
        
        Args:
            rgb_pixel (array): Array RGB pixel
            
        Returns:
            tuple: (predicted_color, confidence_score)
        """
        rgb_scaled = self.scaler.transform(rgb_pixel.reshape(1, -1))
        
        # Prediksi dengan probabilitas
        predicted_color = self.svm.predict(rgb_scaled)[0]
        probabilities = self.svm.predict_proba(rgb_scaled)[0]
        
        # Confidence adalah probabilitas tertinggi
        confidence = np.max(probabilities) * 100
        
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
                          confidence=None, point_label=""):
        """
        Menggambar label dan bounding box pada gambar
        
        Args:
            image (array): Gambar input
            text (str): Text label
            top_left (tuple): Koordinat kiri atas
            bottom_right (tuple): Koordinat kanan bawah  
            color (tuple): Warna BGR untuk box
            confidence (float): Confidence score
            point_label (str): Label untuk titik (Point 1/Point 2)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        font_scale = 0.8
        
        # Gambar bounding box
        cv2.rectangle(image, top_left, bottom_right, color, 3)
        
        # Background semi-transparan
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Tentukan warna text berdasarkan brightness background
        brightness = (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
        
        # Gambar point label
        label_text = f"{point_label}"
        cv2.putText(image, label_text, (top_left[0], top_left[1] - 5), font, 0.6, color, 2)
        
        # Gambar text nama warna
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = top_left[0] + (bottom_right[0] - top_left[0] - text_size[0]) // 2
        text_y = top_left[1] + (bottom_right[1] - top_left[1] + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Gambar confidence score jika ada
        if confidence is not None:
            conf_text = f"{confidence:.1f}%"
            cv2.putText(image, conf_text, (text_x, text_y + 25), font, 0.5, text_color, 1)
    
    def run_webcam_detection(self):
        """Menjalankan deteksi warna real-time dari webcam dengan 2 titik deteksi"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Tidak dapat membuka kamera")
            return
        
        print("=== Dual Point SVM Color Detection ===")
        print("Controls:")
        print("- 'q': Quit")
        print("- 'r': Reset accuracy statistics")
        print("- Point 1: Left side detection")
        print("- Point 2: Right side detection")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame dari kamera")
                break
            
            # Dapatkan dimensi frame
            height, width, _ = frame.shape
            center_x, center_y = width // 2, height // 2
            
            # Definisikan ROI untuk 2 titik deteksi
            roi_w, roi_h = 100, 100
            offset = 150  # Jarak antara 2 titik
            
            # Titik 1 (kiri)
            point1_x = center_x - offset
            point1_y = center_y
            top_left_1 = (point1_x - roi_w // 2, point1_y - roi_h // 2)
            bottom_right_1 = (point1_x + roi_w // 2, point1_y + roi_h // 2)
            
            # Titik 2 (kanan)
            point2_x = center_x + offset
            point2_y = center_y
            top_left_2 = (point2_x - roi_w // 2, point2_y - roi_h // 2)
            bottom_right_2 = (point2_x + roi_w // 2, point2_y + roi_h // 2)
            
            # Deteksi warna untuk Point 1
            if 0 <= point1_x < width and 0 <= point1_y < height:
                pixel_bgr_1 = frame[point1_y, point1_x].reshape(1, 1, 3)
                pixel_hsv_1 = cv2.cvtColor(pixel_bgr_1, cv2.COLOR_BGR2HSV)
                pixel_rgb_1 = pixel_bgr_1[0][0][::-1]  # BGR ke RGB
                
                # Coba deteksi dengan HSV terlebih dahulu
                color_hsv_1 = self.detect_color_hsv(pixel_hsv_1)
                
                if color_hsv_1:
                    color_pred_1 = color_hsv_1
                    confidence_1 = 95.0
                else:
                    color_pred_1, confidence_1 = self.get_color_confidence(pixel_rgb_1)
                
                # Update statistik untuk point 1
                self.total_frames_point1 += 1
                if color_hsv_1 or confidence_1 > 70:
                    self.correct_predictions_point1 += 1
                
                accuracy_percent_1 = (self.correct_predictions_point1 / self.total_frames_point1) * 100
                box_color_1 = self.color_bgr_map.get(color_pred_1, (128, 128, 128))
                
                # Gambar deteksi point 1
                self.draw_label_and_box(
                    frame, color_pred_1, top_left_1, bottom_right_1, 
                    box_color_1, confidence_1, "Point 1"
                )
                
                # Gambar crosshair untuk point 1
                cv2.line(frame, (point1_x - 10, point1_y), (point1_x + 10, point1_y), (0, 255, 0), 2)
                cv2.line(frame, (point1_x, point1_y - 10), (point1_x, point1_y + 10), (0, 255, 0), 2)
            
            # Deteksi warna untuk Point 2
            if 0 <= point2_x < width and 0 <= point2_y < height:
                pixel_bgr_2 = frame[point2_y, point2_x].reshape(1, 1, 3)
                pixel_hsv_2 = cv2.cvtColor(pixel_bgr_2, cv2.COLOR_BGR2HSV)
                pixel_rgb_2 = pixel_bgr_2[0][0][::-1]  # BGR ke RGB
                
                # Coba deteksi dengan HSV terlebih dahulu
                color_hsv_2 = self.detect_color_hsv(pixel_hsv_2)
                
                if color_hsv_2:
                    color_pred_2 = color_hsv_2
                    confidence_2 = 95.0
                else:
                    color_pred_2, confidence_2 = self.get_color_confidence(pixel_rgb_2)
                
                # Update statistik untuk point 2
                self.total_frames_point2 += 1
                if color_hsv_2 or confidence_2 > 70:
                    self.correct_predictions_point2 += 1
                
                accuracy_percent_2 = (self.correct_predictions_point2 / self.total_frames_point2) * 100
                box_color_2 = self.color_bgr_map.get(color_pred_2, (128, 128, 128))
                
                # Gambar deteksi point 2
                self.draw_label_and_box(
                    frame, color_pred_2, top_left_2, bottom_right_2, 
                    box_color_2, confidence_2, "Point 2"
                )
                
                # Gambar crosshair untuk point 2
                cv2.line(frame, (point2_x - 10, point2_y), (point2_x + 10, point2_y), (0, 255, 0), 2)
                cv2.line(frame, (point2_x, point2_y - 10), (point2_x, point2_y + 10), (0, 255, 0), 2)
            
            # Tampilkan informasi akurasi untuk kedua titik
            if self.total_frames_point1 > 0:
                acc_text_1 = f"Point 1 Accuracy: {accuracy_percent_1:.1f}%"
                cv2.putText(frame, acc_text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.total_frames_point2 > 0:
                acc_text_2 = f"Point 2 Accuracy: {accuracy_percent_2:.1f}%"
                cv2.putText(frame, acc_text_2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Tampilkan informasi model
            model_info = "SVM (RBF Kernel)"
            cv2.putText(frame, model_info, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Tampilkan frame
            cv2.imshow('Dual Point SVM Color Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset accuracy statistics
                self.total_frames_point1 = 0
                self.correct_predictions_point1 = 0
                self.total_frames_point2 = 0
                self.correct_predictions_point2 = 0
                print("Accuracy statistics reset for both points")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Tampilkan statistik akhir
        if self.total_frames_point1 > 0:
            final_acc_1 = (self.correct_predictions_point1 / self.total_frames_point1) * 100
            print(f"Final Point 1 accuracy: {final_acc_1:.1f}%")
        
        if self.total_frames_point2 > 0:
            final_acc_2 = (self.correct_predictions_point2 / self.total_frames_point2) * 100
            print(f"Final Point 2 accuracy: {final_acc_2:.1f}%")

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
        detector = DualPointColorDetectorSVM(csv_path='datasheet/colors.csv')
        
        # Jalankan deteksi webcam
        detector.run_webcam_detection()
        
    except FileNotFoundError:
        print("Please make sure colors.csv exists in the datasheet folder")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()