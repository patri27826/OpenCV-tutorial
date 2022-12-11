# OpenCV implementation

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

![interface](/image/1.png?raw=true "User Interface")

## 實作功能

- 手刻背景去除及前景檢測 Background Subtraction
- 運用光流法來實現影片特徵點追蹤 Video Tracking using Optical Flow ( cv2.SimpleBlobDetector, calcOpticalFlowPyrLK )
- 使用OpenCV的aruco，來偵測特定marker，並將圖片任一張圖片縮放嵌入至marker所包圍的區域(cv2.aruco)
- 運用PCA做影像重建

## Read File

```python
# Read File
root = tkinter.Tk()
root.withdraw() #use to hide tkinter window
tempdir = fd.askopenfilename(initialdir = os.getcwd(), title = "Select file", filetypes = (("mp4 files","*.mp4"),("all files","*.*")))
if len(tempdir) > 0:
    print("video Path: %s",tempdir)
self.vidcap = cv2.VideoCapture(tempdir)
self.video_label.setText("video loaded")
```

## 1. Background Subtraction

我們在這次實作中，試著不使用OpenCV的函式 cv2.createbackgroundSubtrator()

### 原理

![image](/image/2.png?raw=true "Image")

簡單來說，就是要我們在前25個frame中，對每一個pixel計算出他們的平均跟標準差，當作我們的背景資訊，再透過計算接下來的frame的每個pixel跟我們前面計算的mean的差距是不是大於5倍的標準差，是的話就是前景，設為白，否的話就是背景，設為黑，也就是下式

$$
⁍
$$

$$
If \space Frame(x , y) - mean < std*5 ,then \space mask(x , y) = 0
$$

```python

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

### Result

![image](/image/3.png?raw=true "Image")

## 2. Video Tracking using calcOpticalFlowPyrLK and SimpleBlobDetector

### 2-1 SimpleBlobDetector on First Frame of Video

### 目標 : 讀取影片的第一個frame並偵測出腳上的斑點，並用田字框起來

![image](/image/4.png?raw=true "Image")

```python
# Setup SimpleBlobDetector parameters.
self.params = cv2.SimpleBlobDetector_Params()

# Change thresholds
self.params.minThreshold = 50
self.params.maxThreshold = 255

# Filter by Area.
self.params.filterByArea = True
self.params.minArea = 35
self.params.maxArea = 90

# Filter by Circularity
self.params.filterByCircularity = True
self.params.minCircularity = 0.8

# Filter by Convexity
self.params.filterByConvexity = True
self.params.minConvexity = 0.8

# Filter by Inertia
self.params.filterByInertia = True
self.params.minInertiaRatio = 0.5

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(self.params)
_,frame = self.vidcap.read()
```

設定SimpleBlobDetector，給予你想要偵測的東西的一些條件我這邊是設定成一個檢測我範例中影片腳上的斑點，具體參數可以參考網路

[https://blog.csdn.net/weixin_44901043/article/details/123275650](https://blog.csdn.net/weixin_44901043/article/details/123275650)

[https://www.796t.com/content/1546661369.html](https://www.796t.com/content/1546661369.html)

設定好後，將參數指定給一個detector做之後的檢測

```python
# Detect blobs.
keypoints = detector.detect(frame)
pts = cv2.KeyPoint.convert(keypoints) 
```

SimpleBlobDetector的dectector.detect所拿到的keypoints的資料型態為cv2.keypoints的格式，會包含更多資訊，但我們在這邊不需要，於是使用他內鍵函式cv2.KeyPoint.convert，可以將cv2.keypoints轉成Point2f格式，即可取得點的x,y座標

```python
for i in range(len(pts)):
    cv2.rectangle(frame, (int(pts[i][0])-6, int(pts[i][1])-6), (int(pts[i][0])+6, int(pts[i][1])+6), (0,255,0), 1, cv2.LINE_AA)
    cv2.line(frame, (int(pts[i][0])-6, int(pts[i][1])), (int(pts[i][0])+6, int(pts[i][1])), (0,255,0), 1, cv2.LINE_AA)
    cv2.line(frame, (int(pts[i][0]), int(pts[i][1])-6), (int(pts[i][0]), int(pts[i][1])+6), (0,255,0), 1, cv2.LINE_AA)
# Draw detected blobs as red circles.
#frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", frame)
#cv2.imshow("Keypoints", frame_with_keypoints)
cv2.waitKey(0)
```

將所有點畫在圖上，我上述的式子會再點的周圍畫出一個田字，如果只想將點圈起來，可以使用註解的部分，註解的cv2.drawKeypoints會回傳畫完keypoints的frame

### Result

![image](/image/5.png?raw=true "Image")

### 2-2 Video Tracking using calcOpticalFlowPyrLK and SimpleBlobDetector

### 目標 : 追蹤腳上斑點的移動軌跡

![image](/image/6.png?raw=true "Image")

```python
# Setup SimpleBlobDetector parameters.
self.params = cv2.SimpleBlobDetector_Params()

# Change thresholds
self.params.minThreshold = 50
self.params.maxThreshold = 255

# Filter by Area.
self.params.filterByArea = True
self.params.minArea = 35
self.params.maxArea = 90

# Filter by Circularity
self.params.filterByCircularity = True
self.params.minCircularity = 0.8

# Filter by Convexity
self.params.filterByConvexity = True
self.params.minConvexity = 0.8

# Filter by Inertia
self.params.filterByInertia = True
self.params.minInertiaRatio = 0.5

# Create a detector with the parameters

detector = cv2.SimpleBlobDetector_create(self.params)
```

一樣要設定SimpleBlobDetector parameters

```python
lk_params = dict( winSize = (15, 15), 
            maxLevel = 2, 
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
```

設定calcOpticalFlowPyrLK的參數，可以參考

[https://www.twblogs.net/a/5b8d6dc02b717718833dea03](https://www.twblogs.net/a/5b8d6dc02b717718833dea03)

[https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#cv2.calcOpticalFlowPyrLK](https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#cv2.calcOpticalFlowPyrLK)

```python
# Create some random colors
color = np.random.randint(0,255,(100,3))
ret, old_frame = self.vidcap.read()

#detect keypoints
detector = cv2.SimpleBlobDetector_create(self.params)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#keypoints to Point2f
keypoints = detector.detect(old_frame)
p0 = cv2.KeyPoint.convert(keypoints)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
```

因為calcOpticalFlowPyrLK只吃Point2F的參數，故要先convert，這邊p0代表前一個frame的keypoints，因為calcOpticalFlowPyrLK要吃的參數是前一個frame跟前一個frame的keypoints以及當前的frame，並會給你當前的frame的keypoints，並且會對應到舊的keypoints，你有了前後frame的keypoints，就可以將他們兩點之間畫線，紀錄點的移動

```python
while True:
      ret, frame = self.vidcap.read()
      if ret == True:
          frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          vis = frame.copy()

          # calculate optical flow
          p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

          # draw the tracks
          for i,(new,old) in enumerate(zip(p1,p0)):
              a,b = new.ravel()
              c,d = old.ravel()
              mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
              frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
          img = cv2.add(frame,mask)

          cv2.imshow('frame',img)
          k = cv2.waitKey(30) & 0xff
          if k == 27:
              break

          # Now update the previous frame and previous points
          old_gray = frame_gray.copy()
          p0 = p1.reshape(-1,1,2)
      else:
          break
```

### Result

![image](/image/7.png?raw=true "Image")

## 3. cv2.aruco and Perspective Transform

### 目標 : 給定一個影片、圖片，將圖片嵌入至影片對應的方框

![image](/image/8.png?raw=true "Image")

### 實作

```python
img_height,img_width = self.Q3Image.shape[:2]
pts2 = np.float32([[0,0], [img_width,0], [img_width,img_height], [0,img_height]])
```

讀取圖片的長寬，並定義好pts為圖片的四個對應點，待會要與偵測到的aruco點做座標轉換，分別為左上、右上、右下、左下四點

```python
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
```

ArUco是一個用來偵測物體尺寸的函式，可以使用它定義好的dictionary去決定你要檢測的物體尺寸，我這邊使用cv2.aruco.DICT_4X4_50去檢測我的四個角的小方框

```python
for (markerCorner, markerID) in zip(corners, ids):
    # 提取標記角（始終按左上角、右上角、右下角和左下角順序返回）
    corners = markerCorner.reshape((4, 2))
    (topLeft, topRight, bottomRight, bottomLeft) = corners
    # 將每個 (x, y) 坐標對轉換為整數
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1])
		#將偵測到的四個對應點填入pts1
    if markerID == 1:  #左上
        pts1[0][0], pts1[0][1] = topLeft
    elif markerID == 2: #右上
        pts1[1][0], pts1[1][1] = topRight
    elif markerID == 3: #右下
        pts1[2][0], pts1[2][1] = bottomRight
    elif markerID == 4: #左下
        pts1[3][0], pts1[3][1] = bottomLeft
```

檢測結果

![image](/image/9.png?raw=true "Image")

```python
trans_matrix = cv2.getPerspectiveTransform(pts2,pts1)
trans_image = cv2.warpPerspective(self.Q3Image, trans_matrix, (len(frame[0]), len(frame)))
```

計算轉換矩陣，並計算出轉換的圖片

原圖

![1.png](OpenCV%20implementation%20d745942a502b4a7d81138568ddc8e560/1.png)

轉換後

![image](/image/10.png?raw=true "Image")

對應影片

![image](/image/11.png?raw=true "Image")

```python
trans_grey = cv2.cvtColor(trans_image,cv2.COLOR_BGR2GRAY)
ret, trans_mask = cv2.threshold(trans_grey, 0, 255, cv2.THRESH_BINARY)
reverse_mask = cv2.bitwise_not(trans_mask)
```

要將轉換後的圖片貼上去，要先把影片的frame內的對應位置挖掉，所以要先製作mask

先將轉換後圖片轉為灰階，透過cv2.threshold可以將圖片轉為圖片部分為全白、其餘全黑，在做bitwise_or，將黑轉白、白轉黑，如圖

![image](/image/12.png?raw=true "Image")

```python
blank = cv2.bitwise_and(frame, frame, mask=reverse_mask)
concate = cv2.add(blank, trans_image)
```

再來就是跟前面解釋的一樣，將mask跟原圖做bitwise_and即可挖空原圖，再透過cv2.add，將兩圖疊加

### Result

![image](/image/13.png?raw=true "Image")

完整程式碼

```python
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()

img_height,img_width = self.Q3Image.shape[:2]
pts2 = np.float32([[0,0], [img_width,0], [img_width,img_height], [0,img_height]])

while True:
    _,frame = self.vidcap.read()
    if frame is None:
        break
    
    # dectoct all aruco markers
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    # at least one marker
    if len(corners) > 0:
        # make ids [3 4 2 1] to [[3] [4] [2] [1]]
        ids = ids.flatten()
        pts1 = np.zeros(shape=(4, 2),dtype=np.float32)

        for (markerCorner, markerID) in zip(corners, ids):
            # 提取標記角（始終按左上角、右上角、右下角和左下角順序返回）
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # 將每個 (x, y) 坐標對轉換為整數
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            #將偵測到的四個對應點填入pts1
            if markerID == 1:
                pts1[0][0], pts1[0][1] = topLeft
            elif markerID == 2:
                pts1[1][0], pts1[1][1] = topRight
            elif markerID == 3:
                pts1[2][0], pts1[2][1] = bottomRight
            elif markerID == 4:
                pts1[3][0], pts1[3][1] = bottomLeft

    # Apply Perspective Transform Algorithm
    trans_matrix = cv2.getPerspectiveTransform(pts2,pts1)
    trans_image = cv2.warpPerspective(self.Q3Image, trans_matrix, (len(frame[0]), len(frame)))
    
    trans_grey = cv2.cvtColor(trans_image,cv2.COLOR_BGR2GRAY)
    ret, trans_mask = cv2.threshold(trans_grey, 0, 255, cv2.THRESH_BINARY)
    reverse_mask = cv2.bitwise_not(trans_mask)

    blank = cv2.bitwise_and(frame, frame, mask=reverse_mask)
    concate = cv2.add(blank, trans_image)

   
    cv2.imshow("Frame", concate)
    key = cv2.waitKey(5) & 0xFF
  
    if key == ord("q"):
        break
```

## 4. PCA - Image Reconstruction

### 目標 : 用PCA做Image Reconstruction

![image](/image/14.png?raw=true "Image")

```python
ig = plt.figure(figsize=(20, 14))
flat_imgs = []
imgs = []
for i in range(1,31):
    path = self.folder+"/sample ("+str(i)+").jpg"
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # (350, 350, 3)
		# image flatten
    imgs.append(img)
    flat = img.flatten()
    flat_imgs.append(flat)
    subplot = None
    if i >= 16:
        subplot = fig.add_subplot(4, 15, i+15)
    else:
        subplot = fig.add_subplot(4, 15, i)
    #Set title
    if i == 16 or i == 1:
            subplot.set_title("origin")
    plt.imshow(img)
```

圖片讀取和plot設定，並將圖片壓成一維陣列

```python
arr = np.array(flat_imgs)
pca = PCA(n_components=27)
pca_reduced = pca.fit_transform(arr)
pca_recovered = pca.inverse_transform(pca_reduced)
# Normalize image to between 0 and 255
pca_recovered = cv2.normalize(pca_recovered, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)指定
```

做PCA，指定n_components為27，並將轉換後的陣列做正規化，因為不保證pixel會在0~255之間

```python
for i in range(0,30):
    subplot = None
    image_pca = np.reshape(pca_recovered[i], (350, 350, 3)).astype('uint8')
    if i >= 15:
        subplot = fig.add_subplot(4, 15, i+31)
    else:
        subplot = fig.add_subplot(4, 15, i+16)
    if i == 15 or i == 0:
            subplot.set_title("reconstruction")
    plt.imshow(image_pca)

    gray_origin = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
    gray_pca = cv2.cvtColor(image_pca, cv2.COLOR_RGB2GRAY)
    self.Reconst_error.append(np.sqrt(np.sum(np.square(np.subtract(gray_origin, gray_pca)))))
plt.show()
```

轉換後圖片輸出到plot上

### Result

![image](/image/15.png?raw=true "Image")