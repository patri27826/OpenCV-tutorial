from ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication
from matplotlib import pyplot as plt
import sys
import cv2
import os
import numpy as np
import tkinter.filedialog as fd
import tkinter
from sklearn.decomposition import PCA

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.BindComponents()
        self.imgs = []
        self.vidcap = None
        self.mean = None
        self.std = None
        self.params = None
        self.Q3Image = None
        self.folder = None
        self.Reconst_error = []

    def BindComponents(self):
        self.load_video.clicked.connect(self.load_video_clk)
        self.load_image.clicked.connect(self.load_image_clk)
        self.load_folder.clicked.connect(self.load_folder_clk)
        self.btn1_1.clicked.connect(self.btn1_1_clk)
        self.btn2_1.clicked.connect(self.btn2_1_clk)
        self.btn2_2.clicked.connect(self.btn2_2_clk)
        self.btn3_1.clicked.connect(self.btn3_1_clk)
        self.btn4_1.clicked.connect(self.btn4_1_clk)
        self.btn4_2.clicked.connect(self.btn4_2_clk)

    def load_video_clk(self):
        root = tkinter.Tk()
        root.withdraw() #use to hide tkinter window
        tempdir = fd.askopenfilename(initialdir = os.getcwd(), title = "Select file", filetypes = (("mp4 files","*.mp4"),("all files","*.*")))
        if len(tempdir) > 0:
            print("video Path: %s",tempdir)
        self.vidcap = cv2.VideoCapture(tempdir)
        self.video_label.setText("video loaded")
        
    def load_image_clk(self):
        root = tkinter.Tk()
        root.withdraw() #use to hide tkinter window
        tempdir = fd.askopenfilename(initialdir = os.getcwd(), title = "Select file", filetypes = (("PNG files","*.png"),("all files","*.*")))
        if len(tempdir) > 0:
            print("image Path: %s",tempdir)
        self.Q3Image = cv2.imread(tempdir)
        self.image_label.setText("image loaded")

    def load_folder_clk(self):
        root = tkinter.Tk()
        root.withdraw() #use to hide tkinter window
        tempdir = fd.askdirectory(parent=root, initialdir = os.getcwd(), title='Please select a directory')
        if len(tempdir) > 0:
            print("folder Path: %s",tempdir)
        self.folder = tempdir
        self.folder_label.setText("folder loaded")

    def btn1_1_clk(self):
        cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
        frames = []
        set = False

        while self.vidcap.isOpened():
            ret, frame = self.vidcap.read()

            if not ret:
                break

            frame = np.array(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert video frames to Gray 
            result  = np.zeros_like(gray)   # all to be 0     
            removebg = np.zeros_like(gray)  # all to be 0   
            img_show = cv2.hconcat([frame, frame, frame])

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
                diff = np.subtract(gray, mean) # subtract
                diff = np.absolute(diff)
                result[diff > (5*std)] = 255 # larger than 5 times standard deviation, set testing pixel to 255 
                gray_three_channel = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                removebg = cv2.bitwise_and(frame, gray_three_channel)
                img_show = cv2.hconcat([frame, gray_three_channel, removebg])
            
            cv2.imshow('Video', img_show)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        self.vidcap.release()    
        cv2.destroyAllWindows()

    def btn2_1_clk(self):
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
        
        # Detect blobs.
        keypoints = detector.detect(frame)
        pts = cv2.KeyPoint.convert(keypoints) 
        print(pts[0][0])
        for i in range(len(pts)):
            cv2.rectangle(frame, (int(pts[i][0])-6, int(pts[i][1])-6), (int(pts[i][0])+6, int(pts[i][1])+6), (0,255,0), 1, cv2.LINE_AA)
            cv2.line(frame, (int(pts[i][0])-6, int(pts[i][1])), (int(pts[i][0])+6, int(pts[i][1])), (0,255,0), 1, cv2.LINE_AA)
            cv2.line(frame, (int(pts[i][0]), int(pts[i][1])-6), (int(pts[i][0]), int(pts[i][1])+6), (0,255,0), 1, cv2.LINE_AA)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        #frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Show keypoints
        cv2.imshow("Keypoints", frame)
        cv2.waitKey(0)

    def btn2_2_clk(self):
        lk_params = dict( winSize = (15, 15), 
            maxLevel = 2, 
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    

        # Create some random colors
        color = np.random.randint(0,255,(100,3))
        ret, old_frame = self.vidcap.read()

        detector = cv2.SimpleBlobDetector_create(self.params)
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        #keypoints to Point2f
        keypoints = detector.detect(old_frame)
        p0 = cv2.KeyPoint.convert(keypoints)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

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
                key = cv2.waitKey(30) & 0xff

                if key == ord("q"):
                    break

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = p1.reshape(-1,1,2)
            else:
                break

    def btn3_1_clk(self):
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

    def btn4_1_clk(self):
        fig = plt.figure(figsize=(20, 14))
        flat_imgs = []
        imgs = []
        for i in range(1,31):
            path = self.folder+"/sample ("+str(i)+").jpg"
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # (350, 350, 3)
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
        
        arr = np.array(flat_imgs)
        #print(arr.shape)
        pca = PCA(n_components=27)
        pca_reduced = pca.fit_transform(arr)
        pca_recovered = pca.inverse_transform(pca_reduced)
        # Normalize image to between 0 and 255
        #pca_recovered = pca_recovered/(pca_recovered.max()/255.0)
        pca_recovered = cv2.normalize(pca_recovered, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
        
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
            
    def btn4_2_clk(self):
        print("Min Error: ", min(self.Reconst_error))
        print("Max Error: ", max(self.Reconst_error))
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())