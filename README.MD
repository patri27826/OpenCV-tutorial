# README

## 前言

在學校的課堂上實做的作業，整理了一下心得用法給之後會用到的人參考

## 使用方法

```python
#usage
# create a conda environment first
pip install -r required.txt
python main.py
```

## 使用者介面

![interface](image\1.png?raw=true "User Interface")

## 實作功能

- 手刻背景去除及前景檢測 Background Subtraction
- 運用光流法來實現影片特徵點追蹤 Video Tracking using Optical Flow ( cv2.SimpleBlobDetector, calcOpticalFlowPyrLK )
- 使用OpenCV的aruco，來偵測特定marker，並將圖片任一張圖片縮放嵌入至marker所包圍的區域(cv2.aruco)
- 運用PCA做影像重建

## 1. Background Subtraction

我們在這次實作中，試著不使用OpenCV的函式 cv2.createbackgroundSubtrator()

### 原理

![image](image\2.png?raw=true "Image")

簡單來說，就是要我們在前25個frame中，對每一個pixel計算出他們的平均跟標準差，當作我們的背景資訊，再透過計算接下來的frame的每個pixel跟我們前面計算的mean的差距是不是大於5倍的標準差，是的話就是前景，設為白，否的話就是背景，設為黑，也就是下式

$$
⁍
$$

$$
If \space Frame(x , y) - mean < std*5 ,then \space mask(x , y) = 0
$$

```python
# Read File
root = tkinter.Tk()
root.withdraw() #use to hide tkinter window
tempdir = fd.askopenfilename(initialdir = os.getcwd(), title = "Select file", filetypes = (("mp4 files","*.mp4"),("all files","*.*")))
if len(tempdir) > 0:
    print("video Path: %s",tempdir)
self.vidcap = cv2.VideoCapture(tempdir)
self.video_label.setText("video loaded")

# Progress
cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
frames = []
set = False  #flag for 25+ frame

while self.vidcap.isOpened():
    ret, frame = self.vidcap.read()
    if not ret:
        break

    frame = np.array(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert video frames to Gray 
    result  = np.zeros_like(gray)   # all to be 0     
    removebg = np.zeros_like(gray)  # all to be 0   
    img_show = cv2.hconcat([frame, frame, frame])  #預先設定顯示圖片為0

    if len(frames) < 25:         
        frames.append(gray)

    elif len(frames) == 25:
        all_frames = np.array(frames)
        mean = np.mean(all_frames, axis=0) #build a gaussian model with mean
        std = np.std(all_frames, axis=0) #build a gaussian model with standard deviation 
        std[std < 5] = 5 # if standard deviation is less then 5, set to 5
        set = True
    if set :
        # after 26th fram
        diff = np.subtract(gray, mean) # 將frame與mean相減
        diff = np.absolute(diff)     
        result[diff > (5*std)] = 255 #大於5*標準差，設為255
        gray_three_channel = cv2.cvtColor(result, cv2.COLOR_GRA2BGR)  #result是灰階，轉回RGB
        removebg = cv2.bitwise_and(frame, gray_three_channel) #與原圖做and
        img_show = cv2.hconcat([frame, gray_three_channel, removebg]) #將原圖、mask、結果合併
    
    cv2.imshow('Video', img_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

這邊解釋一下最後幾行

```python
gray_three_channel = cv2.cvtColor(result, cv2.COLOR_GRA2BGR)
```

將他轉回RGB是因為他現在是灰階，每個pixel只有一個值，轉為灰階才能跟其他照片做and

```python
removebg = cv2.bitwise_and(frame, gray_three_channel) #與原圖做and
```

cv2.bitwise的相關操作可以看這篇 [https://ithelp.ithome.com.tw/articles/10248721](https://ithelp.ithome.com.tw/articles/10248721)

與原圖做bitwise_and的原因是因為我做完與mean,std的比較後，背景是黑(0)，前景為白(255)，你與0做bitwise_and就會變成黑色，與255做bitwise_and就會保留原有值，因此上面式子就可以做到把圖片中的前景提取出來，讓背景變為黑。

## 2. Video Tracking using calcOpticalFlowPyrLK and SimpleBlobDetector