import cv2
import sub

if __name__ == '__main__':
    
    i = sub.imageprocessing(url = 0, capture = True, ui = True)
    i.start() 
    
    while 1 :
        
        img = i.read()
        i.real_high = 100
        print(i.judge(img.copy()))

        if cv2.waitKey(1)==27:
            i.stop() 
            break