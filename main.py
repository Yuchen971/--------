import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import shutil
from string import ascii_uppercase
import sys
import engine

def resource_path(relative_path):
    """ 获取资源的绝对路径 """
    try:
        # PyInstaller 创建临时文件夹 _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("批量图像处理器")
        
        self.tab_control = ttk.Notebook(root)
        
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab1, text='文件夹结构')
        self.tab_control.add(self.tab2, text='自动截头')
        self.tab_control.add(self.tab3, text='自动抠图')
        
        self.tab_control.pack(expand=1, fill='both')
        
        self.create_tab1()
        self.create_tab2()
        self.create_tab3()
        
        # 加载人脸检测模型
        prototxt_path = resource_path('Face Detector Prototxt.prototxt')
        caffemodel_path = resource_path('Face Detection Model.caffemodel')
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        # 加载 U2NET 模型
        # engine.load_model()
    
    def create_tab1(self):
        self.folder_label = ttk.Label(self.tab1, text="选择并处理根文件夹：")
        self.folder_label.pack(pady=10)
        
        self.folder_button = ttk.Button(self.tab1, text="浏览并处理", command=self.browse_and_process_folder)
        self.folder_button.pack(pady=10)
    
    def create_tab2(self):
        self.size_label = ttk.Label(self.tab2, text="目标尺寸 (宽 x 高)：")
        self.size_label.pack(pady=10)
        
        self.size_entry = ttk.Entry(self.tab2)
        self.size_entry.insert(0, "1350x1800")
        self.size_entry.pack(pady=10)
        
        self.dpi_label = ttk.Label(self.tab2, text="DPI (X, Y)：")
        self.dpi_label.pack(pady=10)
        
        self.dpi_entry = ttk.Entry(self.tab2)
        self.dpi_entry.insert(0, "300,300")
        self.dpi_entry.pack(pady=10)
        
        self.confidence_label = ttk.Label(self.tab2, text="置信度：")
        self.confidence_label.pack(pady=10)
        
        self.confidence_entry = ttk.Entry(self.tab2)
        self.confidence_entry.insert(0, "0.5")
        self.confidence_entry.pack(pady=10)
        
        self.percentage_label = ttk.Label(self.tab2, text="截取百分比：")
        self.percentage_label.pack(pady=10)
        
        self.percentage_entry = ttk.Entry(self.tab2)
        self.percentage_entry.insert(0, "50")
        self.percentage_entry.pack(pady=10)
        
        self.batch_button = ttk.Button(self.tab2, text="批量处理", command=self.batch_process)
        self.batch_button.pack(pady=10)
        
        self.single_button = ttk.Button(self.tab2, text="单个处理", command=self.single_process)
        self.single_button.pack(pady=10)
        
        self.image_label = ttk.Label(self.tab2)
        self.image_label.pack(pady=10)
        
        self.download_button = ttk.Button(self.tab2, text="下载", command=self.download_image, state=tk.DISABLED)
        self.download_button.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(self.tab2, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.pack_forget()  # 初始时隐藏进度条
        
        self.progress_label = ttk.Label(self.tab2, text="")
        self.progress_label.pack(pady=5)
        self.progress_label.pack_forget()  # 初始时隐藏进度标签
    
    def create_tab3(self):
        self.size_label3 = ttk.Label(self.tab3, text="目标尺寸 (宽 x 高)：")
        self.size_label3.pack(pady=10)
        
        self.size_entry3 = ttk.Entry(self.tab3)
        self.size_entry3.insert(0, "1350x1800")
        self.size_entry3.pack(pady=10)
        
        self.dpi_label3 = ttk.Label(self.tab3, text="DPI (X, Y)：")
        self.dpi_label3.pack(pady=10)
        
        self.dpi_entry3 = ttk.Entry(self.tab3)
        self.dpi_entry3.insert(0, "300,300")
        self.dpi_entry3.pack(pady=10)
        
        self.batch_button3 = ttk.Button(self.tab3, text="批量处理", command=self.batch_process_matting)
        self.batch_button3.pack(pady=10)
        
        self.single_button3 = ttk.Button(self.tab3, text="单个处理", command=self.single_process_matting)
        self.single_button3.pack(pady=10)
        
        self.image_label3 = ttk.Label(self.tab3)
        self.image_label3.pack(pady=10)
        
        self.download_button3 = ttk.Button(self.tab3, text="下载", command=self.download_image_matting, state=tk.DISABLED)
        self.download_button3.pack(pady=10)
        
        self.progress_bar3 = ttk.Progressbar(self.tab3, orient="horizontal", length=300, mode="determinate")
        self.progress_bar3.pack(pady=10)
        self.progress_bar3.pack_forget()  # 初始时隐藏进度条
        
        self.progress_label3 = ttk.Label(self.tab3, text="")
        self.progress_label3.pack(pady=5)
        self.progress_label3.pack_forget()  # 初始时隐藏进度标签
    
    def browse_and_process_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.process_folder(self.folder_path)
            messagebox.showinfo("成功", "文件夹处理成功")
    
    def process_folder(self, folder_path):
        def get_all_image_files(folder):
            image_files = []
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(root, file))
            return image_files

        def delete_non_directory_files(folder):
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)

        def move_images_to_folder(images, destination_folder, spu_name):
            letter_index = 0
            for image_path in images:
                if letter_index < len(ascii_uppercase):
                    new_name = f"{spu_name}-{ascii_uppercase[letter_index]}"
                else:
                    first_letter = ascii_uppercase[(letter_index // 26) - 1]
                    second_letter = ascii_uppercase[letter_index % 26]
                    new_name = f"{spu_name}-{first_letter}{second_letter}"
                
                extension = os.path.splitext(image_path)[1]
                new_image_path = os.path.join(destination_folder, new_name + extension)
                shutil.move(image_path, new_image_path)
                letter_index += 1

        def delete_empty_folders(folder):
            for root, dirs, files in os.walk(folder, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)

        def delete_folder_and_files(folder):
            for root, dirs, files in os.walk(folder, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    os.rmdir(dir_path)
            os.rmdir(folder)

        def delete_folders_without_images(folder):
            for root, dirs, files in os.walk(folder, topdown=False):
                has_image = any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in files)
                if not has_image and not dirs:
                    delete_folder_and_files(root)

        def process_spu_folder(spu_folder):
            all_items = os.listdir(spu_folder)
            has_subfolders = any(os.path.isdir(os.path.join(spu_folder, item)) for item in all_items)
            if has_subfolders:
                delete_non_directory_files(spu_folder)
                images = get_all_image_files(spu_folder)
                move_images_to_folder(images, spu_folder, os.path.basename(spu_folder))
                delete_empty_folders(spu_folder)
                delete_folders_without_images(spu_folder)
            else:
                images = get_all_image_files(spu_folder)
                move_images_to_folder(images, spu_folder, os.path.basename(spu_folder))

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                process_spu_folder(item_path)

    def batch_process(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        
        size = self.size_entry.get().split('x')
        target_width, target_height = int(size[0]), int(size[1])
        dpi = tuple(map(int, self.dpi_entry.get().split(',')))
        confidence = float(self.confidence_entry.get())
        cut_percentage = float(self.percentage_entry.get()) / 100
        
        # 获取所有子文件夹中的图片文件
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        total_files = len(image_files)
        
        # 创建进度条弹窗
        progress_window = tk.Toplevel(self.root)
        progress_window.title("处理进度")
        progress_window.geometry("300x100")
        
        progress_label = ttk.Label(progress_window, text="正在处理图片...")
        progress_label.pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=250, mode="determinate")
        progress_bar.pack(pady=10)
        
        progress_bar["maximum"] = total_files
        progress_bar["value"] = 0
        
        for i, image_path in enumerate(image_files, 1):
            self.process_image(image_path, target_width, target_height, dpi, confidence, cut_percentage)
            
            progress_bar["value"] = i
            progress_label.config(text=f"处理进度: {i}/{total_files}")
            progress_window.update()
        
        progress_window.destroy()  # 关闭进度条弹窗
        messagebox.showinfo("成功", "批量处理完成")
    
    def single_process(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        size = self.size_entry.get().split('x')
        target_width, target_height = int(size[0]), int(size[1])
        dpi = tuple(map(int, self.dpi_entry.get().split(',')))
        confidence = float(self.confidence_entry.get())
        cut_percentage = float(self.percentage_entry.get()) / 100
        
        processed_image = self.process_image(file_path, target_width, target_height, dpi, confidence, cut_percentage, display=True)
        
        # 移除了对 processed_image 是否为 None 的检查
        self.display_image(processed_image)
        self.download_button.config(state=tk.NORMAL)
        self.processed_image = processed_image

    def process_image(self, image_path, target_width, target_height, dpi, confidence, cut_percentage, display=False):
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误：无法读取图像 {image_path}")
            return None

        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        face_detected = False
        for i in range(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > confidence:
                face_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                chin_y = startY + int(cut_percentage * (endY - startY))
                cropped_img = img[chin_y:, :]
                break

        if not face_detected:
            print(f"警告：在图像 {image_path} 中未检测到人脸")
            # 如果未检测到人脸，使用整个图像
            cropped_img = img

        # 无论是否检测到人脸，都调整图像大小
        resized_img = cv2.resize(cropped_img, (target_width, target_height))
        
        if display:
            return Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        else:
            output_path = os.path.splitext(image_path)[0] + '_processed.jpg'
            pil_image = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
            pil_image.save(output_path, 'JPEG', dpi=dpi, quality=100)
        return None
    
    def display_image(self, image):
        # 计算缩放比例，预览图的最大边长400像素
        max_size = 400
        ratio = max_size / max(image.width, image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        
        # 缩放图像
        preview_image = image.copy()
        preview_image.thumbnail(new_size, Image.LANCZOS)
        
        photo = ImageTk.PhotoImage(preview_image)
        self.image_label3.config(image=photo)  # 更改为 image_label3
        self.image_label3.image = photo
    
    def download_image(self):
        if hasattr(self, 'processed_image'):
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
            if file_path:
                dpi = tuple(map(int, self.dpi_entry.get().split(',')))
                self.processed_image.save(file_path, 'JPEG', dpi=dpi, quality=95)
                messagebox.showinfo("成功", "图像已保存")
        else:
            messagebox.showerror("错误", "没有可下载的图像")
    
    def single_process_matting(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        size = self.size_entry3.get().split('x')
        target_width, target_height = int(size[0]), int(size[1])
        dpi = tuple(map(int, self.dpi_entry3.get().split(',')))
        
        processed_image = self.process_image_matting(file_path, target_width, target_height, dpi, display=True)
        
        if processed_image:
            self.display_image(processed_image)
            self.download_button3.config(state=tk.NORMAL)
            self.processed_image = processed_image
        else:
            messagebox.showerror("错误", "图像处理失败")

    def batch_process_matting(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        
        size = self.size_entry3.get().split('x')
        target_width, target_height = int(size[0]), int(size[1])
        dpi = tuple(map(int, self.dpi_entry3.get().split(',')))
        
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        total_files = len(image_files)
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("处理进度")
        progress_window.geometry("300x100")
        
        progress_label = ttk.Label(progress_window, text="正在处理图片...")
        progress_label.pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=250, mode="determinate")
        progress_bar.pack(pady=10)
        
        progress_bar["maximum"] = total_files
        progress_bar["value"] = 0
        
        for i, image_path in enumerate(image_files, 1):
            self.process_image_matting(image_path, target_width, target_height, dpi, overwrite=True)
            
            progress_bar["value"] = i
            progress_label.config(text=f"处理进度: {i}/{total_files}")
            progress_window.update()
        
        progress_window.destroy()
        messagebox.showinfo("成功", "批量处理完成")

    def process_image_matting(self, image_path, target_width, target_height, dpi, display=False, overwrite=False):
        img = Image.open(image_path)
        if img is None:
            print(f"错误：无法读取图像 {image_path}")
            return None

        # 使用 remove_bg_mult 进行抠图
        processed_img = engine.remove_bg_mult(img)

        # 调整图像大小
        resized_img = processed_img.resize((target_width, target_height), Image.LANCZOS)
        
        if display:
            return resized_img
        else:
            if overwrite:
                output_path = image_path
            else:
                output_path = os.path.splitext(image_path)[0] + '_matted.jpg'
            resized_img.save(output_path, 'JPEG', dpi=dpi, quality=95)
        return None

    def download_image_matting(self):
        if hasattr(self, 'processed_image'):
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
            if file_path:
                dpi = tuple(map(int, self.dpi_entry3.get().split(',')))
                self.processed_image.save(file_path, 'JPEG', dpi=dpi, quality=95)
                messagebox.showinfo("成功", "图像已保存")
        else:
            messagebox.showerror("错误", "没有可下载的图像")

    def process_single_image(self):
        # ... 现有代码 ...
        
        # 在处理完成后添加以下代码
        if os.path.exists(output_path):
            # 打开处理后的图像
            processed_img = Image.open(output_path)
            # 调整图像大小以适应显示
            processed_img.thumbnail((300, 300))
            # 转换为 PhotoImage 对象
            photo = ImageTk.PhotoImage(processed_img)
            
            # 创建新窗口显示结果
            result_window = tk.Toplevel(self.master)
            result_window.title("处理结果")
            
            # 创建 Label 来显示图像
            img_label = tk.Label(result_window, image=photo)
            img_label.image = photo  # 保持对图像的引用
            img_label.pack()
            
            # 添加文本说明
            tk.Label(result_window, text="处理完成！").pack()
        else:
            messagebox.showerror("错误", "处理失败，未生成输出图像。")
        
        # ... 现有代码 ...


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()