from os.path import exists

import glob
import cv2
import numpy as np
import time
import copy
from datetime import datetime
import os

def FName():
    now = datetime.now()
    #date_time_str = now.strftime("%Y-%m-%d_%H%M%S%f")
    date_time_str = now.strftime("%Y-%m-%d_%H%M%S")
    return date_time_str 

def LoadMatriksKamera(NoKamera):
    P=[];
    sf1 =str(NoKamera)+"_K.csv"
    sf2 =str(NoKamera)+"_MatrixTransformasi.csv"
    
    f = exists(sf1)&  exists(sf2)
    if f:
        #Load Parameter Intrisk
        K=np.loadtxt(sf1,delimiter=",")
        I=np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        #Load Matrix Transformasi
        RT=np.loadtxt(sf2,delimiter=",")
        K=np.matmul(K,I);
        P=np.matmul(K,RT)
    return f,P,K,RT

def Triangulasi(x1citra,x2citra,P1,P2):
    #Parameter :
    #  x1citra,x2citra : Pasangan titik citra 2D yang diperoleh dari kamera 1 dan kamera 2
    #  P1,P2 : Matrix proyeksi (3 x 4) kamera 1 dan kamera 2 
    JumlahTitik= x1citra.shape[1];

    #=======================================================
    #1. Melaksanakan Triangulasi
    A=np.zeros((4,4))
    X=[];
    for i in range(JumlahTitik):
        x1 =x1citra[0, i] 
        y1 =x1citra[1, i] 
        x2 =x2citra[0, i] 
        y2 =x2citra[1, i] 
        A =np.array( [x1 * P1[2, :] - P1[0, :],
                        y1 * P1[2, :] - P1[1,:],
                        x2 * P2[2,:] - P2[0,:],
                        y2 * P2[2,:] - P2[1,:]])
        U,S,V =np.linalg.svd(A)
        V=V.transpose()
        X.append(V[:,3])
    X=np.array(X).transpose()
    #   Titik 3D di peroleh adalah matrix (4 x n) berisi koordinat homogen 
    #   hasil triangulasi dengan bentuk sbb:
    #         |w1*x1, w2*x2, ...,wn*xn|
    #   X   = |w1*y1, w2*y2, ...,wn*yn|
    #         |w1*z1, w2*z2, ...,wn*zn|
    #         |w1   ,  w2  , ...,wn   |
    
    #2. Melaksanakan pensekalaan untuk memperoleh koordinat 3D (x_i,y_i,z_i) 
    #   dengan membagi titik homogen dengan bobot yang tersimpan pada baris ke 3
    for i in range(4):
        X[i, :] = X[i, :] / X[3, :]
    #         |x1, x2, ...,xn|
    #   X   = |y1, y2, ...,yn|
    #         |z1, z2, ...,zn|
    #         |1 ,  1, ...,1 |
    
    #3. Mengebalikan dari Koordinat homogen ke koordinat 3D semula. 

    #         |x1, x2, ...,xn|
    #   X   = |y1, y2, ...,yn|
    #         |z1, z2, ...,zn|
    
    return X;

def Kalibrasi(NoKamera, bSaveAllImage=True):
    print(bSaveAllImage)
    baris = 6
    kolom= 8
    # Sepertinya bagian ini merupakan 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)\
    objp = np.zeros((baris*kolom,3), np.float32)
    objp[:,:2] = np.mgrid[0:baris,0:kolom].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    
    objpoints = []
    imgpointsL = []
    imgpointsR = []

    # define a video capture object
    vid = cv2.VideoCapture(NoKamera,cv2.CAP_DSHOW)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 240*2)
    DirName= str(NoKamera)+"-"+FName()
    if bSaveAllImage :
        os.mkdir(DirName)
    l=[]
    tic = time.perf_counter()
    c=0
    while(True):
    	
    	# Capture the video frame
    	# by frame
        ret, frame = vid.read()
        if not(ret):
            break
        frame2 = copy.copy(frame)
        # Memperoleh tinggi dan lebar frame
        height, width, _ = frame.shape
        
        # Bagi frame menjadi dua bagian secara horizontal
        frame_kiri = frame[:, :width // 2]
        frame_kanan = frame[:, width // 2:]
        grayL = cv2.cvtColor(frame_kiri, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frame_kanan, cv2.COLOR_BGR2GRAY)
        retL, cornersL = cv2.findChessboardCorners(grayL, (baris,kolom), cv2.CALIB_CB_FAST_CHECK )
        retR, cornersR = cv2.findChessboardCorners(grayR, (baris,kolom), cv2.CALIB_CB_FAST_CHECK )
        if (retL and retR):
            toc = time.perf_counter()
            if (toc-tic>1):
                cornersL = cv2.cornerSubPix(grayL,cornersL, (11,11), (-1,-1), criteria)
                cornersR = cv2.cornerSubPix(grayR,cornersR, (11,11), (-1,-1), criteria)
                tic = time.perf_counter()
                c=c+1
                objpoints.append(objp)
                imgpointsL.append(cornersL)
                imgpointsR.append(cornersR)
                tic = time.perf_counter()
                sf =DirName+"/"+FName()+"-"+str(c)
                l.append(sf)
                if bSaveAllImage:
                    nama_file_kiri = f'stereoLeft/imageL{str(c)}.png'
                    nama_file_kanan = f'stereoRight/imageR{str(c)}.png'
                    cv2.imwrite(nama_file_kiri, frame_kiri)
                    cv2.imwrite(nama_file_kanan, frame_kanan)
            cv2.drawChessboardCorners(frame_kiri, (baris,kolom), cornersL, retL)
            cv2.drawChessboardCorners(frame_kanan, (baris,kolom), cornersR, retR)
        cv2.putText(frame_kiri,str(c),(50,50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_kanan,str(c),(50,50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)  
        # Tampilkan frame yang telah dibagi dalam dua bagian
        cv2.imshow('Frame Kiri', frame_kiri)
        cv2.imshow('Frame Kanan', frame_kanan)
        ch= cv2.waitKey(1) & 0xFF 
        
        if ch == ord('q'):
            
            break
        if ch == ord('Q'):
            
            break
    
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    mtxL=[]
    distL=[]
    rvecsL=[]
    tvecsL=[]
    mtxR=[]
    distR=[]
    rvecsR=[]
    tvecsR=[]
    snoKamera=str(NoKamera)
    if len(l)>0:
        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
        retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
        np.savetxt(snoKamera+"_KL.csv", mtxL, delimiter=",")
        np.savetxt(snoKamera+"_KR.csv", mtxR, delimiter=",")
        np.savetxt(snoKamera+"_DistorsiL.csv", distL, delimiter=",")
        np.savetxt(snoKamera+"_DistorsiR.csv", distR, delimiter=",")
        if bSaveAllImage:
            np.savetxt(DirName+"/K.csv", mtxL, delimiter=",")
            np.savetxt(DirName+"/K.csv", mtxR, delimiter=",")
            np.savetxt(DirName+"/DistorsiL.csv", distL, delimiter=",")
            np.savetxt(DirName+"/DistorsiR.csv", distR, delimiter=",")
            for i in range(len(rvecsL and rvecsR)):
                np.savetxt(l[i]+"_rvecsL.csv", rvecsL[i], delimiter=",")
                np.savetxt(l[i]+"_rvecsR.csv", rvecsR[i], delimiter=",")
                rotation_matL, _ = cv2.Rodrigues(rvecsL[i])
                rotation_matR, _ = cv2.Rodrigues(rvecsR[i])
                np.savetxt(l[i]+"_rotL.csv", rotation_matL, delimiter=",")
                np.savetxt(l[i]+"_rotR.csv", rotation_matR, delimiter=",")
                np.savetxt(l[i]+"_tvecsL.csv", tvecsL[i], delimiter=",")
                np.savetxt(l[i]+"_tvecsR.csv", tvecsR[i], delimiter=",")
                mL=np.eye(4)
                mL[0:3,0:3]=rotation_matL[0:3,0:3];
                mL[0:3,3]=tvecsL[i][0:3,0];
                np.savetxt(l[i]+"_mL.csv", mL, delimiter=",")
                mR=np.eye(4)
                mR[0:3,0:3]=rotation_matR[0:3,0:3];
                mR[0:3,3]=tvecsR[i][0:3,0];
                np.savetxt(l[i]+"_mL.csv", mR, delimiter=",")
                
    return  retL, mtxL, distL, rvecsL, tvecsL , retR, mtxR, distR, rvecsR, tvecsR        

def MatrixOrientasiKamera(NoKamera1,NoKamera2):
    NoKamera =(NoKamera1,NoKamera2)
    baris = 4
    kolom=6  
    objp = np.zeros((baris*kolom,3), np.float32)
    objp[:,:2] = np.mgrid[0:baris,0:kolom].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    
    vid1 = cv2.VideoCapture(NoKamera1,cv2.CAP_DSHOW)
    vid2 = cv2.VideoCapture(NoKamera2,cv2.CAP_DSHOW)
    #vid1 = cv2.VideoCapture(0)
    #vid2 = cv2.VideoCapture(1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    tic = time.perf_counter()
    corners1=[]
    corners2=[]
    while(True):
        toc = time.perf_counter()
    	
    	# Capture the video frame
    	# by frame
        ret1, frame1 = vid1.read()
        ret2, frame2 = vid2.read()
        
        if not(ret1):
            break
        if not(ret2):
            break
        
        #frame =np.append(frame1,frame2,axis=1)
        frame1a = copy.copy(frame1)
        frame2a = copy.copy(frame2)
      
        gray1 = cv2.cvtColor(frame1a, cv2.COLOR_BGR2GRAY)
        
        ret, corners1 = cv2.findChessboardCorners(gray1, (baris,kolom), cv2.CALIB_CB_FAST_CHECK )
        if ret:
            gray2 = cv2.cvtColor(frame2a, cv2.COLOR_BGR2GRAY)
            ret, corners2 = cv2.findChessboardCorners(gray2, (baris,kolom), cv2.CALIB_CB_FAST_CHECK )
            
        if ret:
            cv2.drawChessboardCorners(frame1a, (kolom,baris), corners1, ret)
            cv2.drawChessboardCorners(frame2a, (kolom,baris), corners2, ret)
            if (toc-tic>1):
                tic = time.perf_counter()
                toc = time.perf_counter()
                corners1 = cv2.cornerSubPix(gray1,corners1, (11,11), (-1,-1), criteria)
                corners2 = cv2.cornerSubPix(gray2,corners2, (11,11), (-1,-1), criteria)
                objpoints = [] # 3d point in real world space
                imgpoints = [] # 2d points in image plane.
                objpoints.append(objp)
                imgpoints.append(corners1)
                objpoints.append(objp)
                imgpoints.append(corners2)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray1.shape[::-1], None, None)
                if ret:
                    for i in range(len(NoKamera)) :
                        sNoKamera = str(NoKamera[i])
                        rotation_mat, _ = cv2.Rodrigues(rvecs[i])    
                        m=np.eye(4)
                        m[0:3,0:3]=rotation_mat[0:3,0:3];
                        m[0:3,3]=tvecs[i][0:3,0];
                        np.savetxt(sNoKamera+"_MatrixTransformasi.csv", m, delimiter=",")
                       
                                               
        fr = np.append(frame1a,frame2a,axis=1)
        cv2.imshow('frame', fr)

        ch= cv2.waitKey(1) & 0xFF 
        
        if ch == ord('q'):
            break
        
        if ch == ord('Q'):
            break
        
    # After the loop release the cap object
    vid1.release()
    vid2.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    return corners1,corners2

def DeteksiMarker(frame1):
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
     # Mendefinisikan Batas warna biru pada model warna HSV
    #Menentukan Batas Hue 
    HueMin =100 
    HueMax =140 
    #Menentukan Batas Saturasi
    SaturasiMin =100 
    SaturasiMax =255 
    #Menentukan Batas Value
    ValueMin=50
    Valuemax=255 
       
    WarnaBatasBawah = np.array([HueMin,SaturasiMin,ValueMin])
    WarnaBatasAtas= np.array([HueMax,SaturasiMax,Valuemax])
    
    # Threshold  citra HSV image untuk memperoleh warna rambu 
    # dengan warna biru 
    mask = cv2.inRange(hsv, WarnaBatasBawah, WarnaBatasAtas)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    JumlahLabel, label = cv2.connectedComponents(mask)
    l=[];
    print("fr")
    r =0
    pd = [];
    for i in range (1,JumlahLabel):
        [bb,cc]=np.where(label == i)
        bmax=(np.max(bb))
        bmin=(np.min(bb))
        cmax=(np.max(cc))
        cmin=(np.min(cc))
        p1=(cmin,bmin)
        p2=(cmax,bmax)
    
        x=(cmin+cmax)/2
        y=(bmin+bmax)/2
        p=(x,y,p1,p2)
        db =bmax-bmin
        dc=cmax -cmin
        if db*dc>r:
            pd=(p1,p2,(x,y))
    #frame =np.append(frame1,frame2,axis=1)
    x =[]
    y =[];
    if len(pd)>0:
        pp = pd[2]
        x=pp[0]
        y=pp[1]
    return x,y,pd;

def Capture(NoKamera1,NoKamera2):
       
    vid1 = cv2.VideoCapture(NoKamera1,cv2.CAP_DSHOW)
    vid2 = cv2.VideoCapture(NoKamera2,cv2.CAP_DSHOW)
    nRes = 1
    Res=((2560,720), (640,480),(320,240))
    w=Res[nRes][0];
    h=Res[nRes][1];
    vid1.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    vid1.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
    vid2.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    vid2.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
    f,P1,K,RT = LoadMatriksKamera(NoKamera1)
    f,P2,K,RT = LoadMatriksKamera(NoKamera2)
        
    #vid1 = cv2.VideoCapture(0)
    #vid2 = cv2.VideoCapture(1)
    
    while(True):
    	# Capture the video frame
    	# by frame
        ret1, frame1 = vid1.read()
        ret2, frame2 = vid2.read()
        
        if not(ret1):
            break
        if not(ret2):
            break     
        #frame =np.append(frame1,frame2,axis=1)
        frame1a = copy.copy(frame1)
        frame2a = copy.copy(frame2)
        b1 = False
        b2 = False 
       
        x1,y1,pd1=DeteksiMarker(frame1a)
        if len(pd1)>0:
            b1 =True 
            cv2.circle(frame1a,(np.int(x1), np.int(y1)), 5, (0,255,0),-1)
        x2,y2,pd2=DeteksiMarker(frame2a)
        if len(pd2)>0:
            b2 = True 
            cv2.circle(frame2a,(np.int(x2), np.int(y2)), 5, (0,255,0),-1)
        if (b1&b2): 
            pc1 =np.array([[x1],[y1]])
            pc2 =np.array([[x2],[y2]])
            
            X = Triangulasi(pc1,pc2,P1,P2)
            print(X.shape)
     
            
            #print(X)
        fr = np.append(frame1a,frame2a,axis=1)
        cv2.imshow('frame', fr)
        
        ch= cv2.waitKey(1) & 0xFF 
        
        if ch == ord('q'):
            break
        
        if ch == ord('Q'):
            break
        
    # After the loop release the cap object
    vid1.release()
    vid2.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

def Calibrating_s(filename):    
    print("Calibrating..")
    baris = 6
    kolom= 8
    square_size = 0.03

    # Persiapkan objek yang dibutuhkan
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Array untuk menyimpan titik-titik 3D dunia
    obj_points = []

    # Array untuk menyimpan titik-titik 2D di gambar
    img_points_left = []
    img_points_right = []



    # Generate koordinat dunia 3D (koordinat objek catur)
    objp = np.zeros((kolom * baris, 3), np.float32)
    objp[:, :2] = np.mgrid[0:kolom, 0:baris].T.reshape(-1, 2) * square_size

    # Ambil daftar gambar untuk kalibrasi
    images_left = glob.glob(r'2024-old-training-model/0-2024-06-06_190014/stereoLeft/imageL*.png')  # Ganti dengan pola nama gambar di direktori Anda
    images_right = glob.glob(r'2024-old-training-model/0-2024-06-06_190014/stereoRight/imageR*.png')  # Ganti dengan pola nama gambar di direktori Anda

    # obj_points, img_points_left, img_points_right,
    # gray_left.shape, K=None, distCoeffs1=None, K2=None, distCoeffs2=None)

    for i in range(len(images_left)):
        img_left = cv2.imread(images_left[i])
        img_right = cv2.imread(images_right[i])

        # Ubah gambar menjadi skala abu-abu
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Temukan titik sudut pada pola catur
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (kolom, baris), None, )
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, (kolom, baris), None)

        if ret_left and ret_right:
            # Refine corner positions
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            # Tambahkan titik 3D dan 2D ke array
            obj_points.append(objp)
            img_points_left.append(corners_left)
            img_points_right.append(corners_right)

    # Ambil informasi kamera
    ret_left, K_left, D_left, rvecs_left, tvecs_left        = cv2.calibrateCamera(obj_points, img_points_left, gray_left.shape[::-1], None, None)
    ret_right, K_right, D_right, rvecs_right, tvecs_right   = cv2.calibrateCamera(obj_points, img_points_right, gray_right.shape[::-1], None, None)

    # Lakukan kalibrasi kamera stereo
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # ret, mtx_left, dist_left, mtx_right, dist_right
    ret, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_left, img_points_right,
        K_left, D_left, K_right, D_right,
        gray_left.shape[::-1], criteria=criteria_stereo, flags=flags
    )
    
    print("ret: \n",ret, "\nKL: \n", K_left, "\nDL: \n", D_left, 
            "\nKR: \n", K_right, "\nDR: \n", D_right, "\nR: \n", R, 
            "\nT: \n", T, "\nE: \n", E, "\nF: \n", F)

    # Simpan parameter kalibrasi untuk digunakan selanjutnya
    np.savez(filename, K_left=K_left, D_left=D_left, K_right=K_right, D_right=D_right, R=R, T=T, E=E, F=F)

    print("Kalibrasi selesai. Parameter disimpan di '%s'" % (filename))
    
        
def StereoCal(n, bSaveAllImage=True, bDoCapture=True, filename="stereo_calibration.npz"):
    print("[STATUS] Arguments [save, capture]:", bSaveAllImage, bDoCapture)
    
    if not bDoCapture:
        Calibrating_s(filename)
        return
    
    baris = 6
    kolom= 8
    # sepertinya criteria mengandung satuan panjang setiap kotak dalam catur
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)\
    objp = np.zeros((baris*kolom,3), np.float32)
    objp[:,:2] = np.mgrid[0:baris,0:kolom].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    
    objpoints = []
    imgpointsL = []
    imgpointsR = []

    # define a video capture object
    cap = cv2.VideoCapture(n, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*2)    
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)    

    DirName= str(n) + "-" + FName()
    
    if bSaveAllImage :
        os.mkdir(DirName)
        os.chdir(DirName)
        os.mkdir("stereoRight")
        os.mkdir("stereoLeft")
    l=[]
    tic = time.perf_counter()
    c=0
    i=0
    while True:
        # Capture the video frame
    	# by frame
        ret, frame = cap.read()
        if not(ret):
            print("[STATUS]:\tCamera not detected\n")
            break
        frame2 = copy.copy(frame)
        # Memperoleh tinggi dan lebar frame
        height, width, _ = frame.shape
        
        # Bagi frame menjadi dua bagian secara horizontal
        frame_kiri = frame[:, :width // 2]
        frame_kanan = frame[:, width // 2:]
        grayL = cv2.cvtColor(frame_kiri, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frame_kanan, cv2.COLOR_BGR2GRAY)
        retL, cornersL = cv2.findChessboardCorners(grayL, (baris,kolom), cv2.CALIB_CB_FAST_CHECK )
        retR, cornersR = cv2.findChessboardCorners(grayR, (baris,kolom), cv2.CALIB_CB_FAST_CHECK )
        if (retL and retR):
            toc = time.perf_counter()
            if (toc-tic>1):
                cornersL = cv2.cornerSubPix(grayL,cornersL, (11,11), (-1,-1), criteria)
                cornersR = cv2.cornerSubPix(grayR,cornersR, (11,11), (-1,-1), criteria)
                tic = time.perf_counter()
                c=c+1
                objpoints.append(objp)
                imgpointsL.append(cornersL)
                imgpointsR.append(cornersR)
                tic = time.perf_counter()

                if bSaveAllImage:
                    nama_file_kiri = f'stereoLeft/imageL{str(c)}.png'
                    nama_file_kanan = f'stereoRight/imageR{str(c)}.png'
                    cv2.imwrite(nama_file_kiri, frame_kiri)
                    cv2.imwrite(nama_file_kanan, frame_kanan)
            cv2.drawChessboardCorners(frame_kiri,  (baris,kolom), cornersL, retL)
            cv2.drawChessboardCorners(frame_kanan, (baris,kolom), cornersR, retR)
        cv2.putText(frame_kiri, str(c),(50,50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_kanan,str(c),(50,50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)  
        # Tampilkan frame yang telah dibagi dalam dua bagian
        cv2.imshow('Frame Kiri', frame_kiri)
        cv2.imshow('Frame Kanan', frame_kanan)
        ch= cv2.waitKey(1) & 0xFF 
        
        if ch == ord('q'):
            break
        if ch == ord('Q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def calibrating_m(filename):
    print("Calibrating..")
    import glob
    # Define the chessboard size
    baris = 6
    kolom= 8

    # Define the criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...,(8,5,0)
    objp = np.zeros((kolom * baris, 3), np.float32)
    objp[:, :2] = np.mgrid[0:kolom, 0:baris].T.reshape(-1, 2)

    # Arrays to store object points and image points from all images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    # Load images
    images = glob.glob('0-2024-06-02_133310\monoCal/*.png')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (kolom, baris), None)
        
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (kolom, baris), corners2, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(100)
    cv2.destroyAllWindows()

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    # Save the camera calibration result for later use
    np.savez(filename, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    pass

def MonoCal(n,  bSaveAllImage=True, bDoCapture=True, filename="mono_calibration.npz"):
    print("[STATUS] Arguments [save, capture]:", bSaveAllImage, bDoCapture)
    
    if not bDoCapture:
        calibrating_m(filename)
        return
    
    baris = 6
    kolom= 8
    # sepertinya criteria mengandung satuan panjang setiap kotak dalam catur
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)\
    objp = np.zeros((baris*kolom,3), np.float32)
    objp[:,:2] = np.mgrid[0:baris,0:kolom].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    
    objpoints = []
    imgpoints = []

    # define a video capture object
    cap = cv2.VideoCapture(n, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240*2)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)9

    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    ret, framel = cap.read()
    print(framel.shape)

    DirName= str(n) + "-" + FName()
    
    if bSaveAllImage :
        os.mkdir(DirName)
        os.chdir(DirName)
        os.mkdir("monoCal")
    l=[]
    tic = time.perf_counter()
    c=0
    i=0
    while True:
        # Capture the video frame
    	# by frame
        ret, framel = cap.read()
        if not(ret):
            print("[STATUS]:\tCamera not detected\n")
            break
        # Memperoleh tinggi dan lebar frame
        height, width, _ = framel.shape
        frame = framel[:, :width // 2]

        # Bagi frame menjadi dua bagian secara horizontal
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (baris,kolom), cv2.CALIB_CB_FAST_CHECK )
        if (ret):
            toc = time.perf_counter()
            if (toc-tic>1):
                corners = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                tic = time.perf_counter()
                c=c+1
                objpoints.append(objp)
                imgpoints.append(corners)
                tic = time.perf_counter()

                if bSaveAllImage:
                    nama_file = f'monoCal/image_{str(c)}.png'
                    cv2.imwrite(nama_file, frame)
            cv2.drawChessboardCorners(frame, (baris,kolom), corners, ret)
        cv2.putText(frame,str(c),(50,50), font, 1, (255, 255, 255), 2, cv2.LINE_AA) 

        # Tampilkan frame yang telah dibagi dalam dua bagian
        cv2.imshow('Frame', frame)
        ch= cv2.waitKey(1) & 0xFF 
        
        if ch == ord('q'):
            break
        if ch == ord('Q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#Program Utama
#ret, mtx, dist, rvecs, tvecs = Kalibrasi(2)
    
c=""
while (True):
    print('1. Kalibrasi Kamera')
    print('2. Menghitung Matriks Orientasi kamera')
    print('3. Menghitung Triangulasi')
    print('4. Capture')
    print('5. Kalibrasi stereo')
    print('6. Kalibrasi mono')

    print('9. Keluar')
    
    c= input("Masukan:")
    
    if c=="9":
        break
    if c=="1":
        print("Kalibrasi Parameter Kamera\n")
        c1= input("Masukan No Kamera :\n")
        nokamera= int(c1)
        Kalibrasi(nokamera)
        
    if c=="2":
        print("Mencari Matriks Orientasi kamera\n")
        c1= input("Masukan No Kamera Ke 1 :\n")
        NoKamera1= int(c1)
        c2= input("Masukan No Kamera Ke 2 :\n")
        NoKamera2= int(c2)
        MatrixOrientasiKamera(NoKamera1,NoKamera2)
    if c=="4":
        c1= input("Masukan No Kamera Ke 1 :\n")
        NoKamera1= int(c1)
        c2= input("Masukan No Kamera Ke 2 :\n")
        NoKamera2= int(c2)
        Capture(c1,c2)
    if c=="5":
        print("Kalibrasi menggunakan kamera stereo")
        c1= input("Masukkan No Kamera:\n")
        NoKamera1= int(c1)
        # c2= input("Ambil Gambar (True) / Kalibrasi(False)? \n")
        # mode = bool(c2)
        filename = "./data-extraction/Kalibrasi/New-Kalibrasi-07052026.npz"
        StereoCal(NoKamera1, bDoCapture=False, filename=filename)
    if c=="6":
        print("kalibrasi monocular camera")
        c1= input("Masukkan no kamera:\n")
        noKamera= int(c1)
        filename = "Kalibrasi/Mono-Kalibrasi-06022024.npz"
        MonoCal(noKamera, bDoCapture=False, filename=filename) 