import cv2
import numpy as np


fourcc = cv2.VideoWriter_fourcc(*'XVID')
font = cv2.FONT_HERSHEY_SIMPLEX
out = cv2.VideoWriter('experiment/pn_compare.avi',fourcc, 15.0, (int(1600*0.8),int(740*0.8)))

for index in range(100, 320):
    pn_3d = cv2.imread('experiment/imgs/pointnet/%d_3d.png' % (index))
    pn_sem = cv2.imread('experiment/imgs/pointnet/%d_sem.png' % (index))
    pn2_3d = cv2.imread('experiment/imgs/pointnet2/%d_3d.png' % (index))
    pn2_sem = cv2.imread('experiment/imgs/pointnet2/%d_sem.png' % (index))

    pn_3d = pn_3d[160:650]
    pn2_3d = pn2_3d[160:650]

    pn_sem = cv2.resize(pn_sem, (800, 250))
    pn2_sem = cv2.resize(pn2_sem, (800, 250))

    pn = np.vstack((pn_3d, pn_sem))
    pn2 = np.vstack((pn2_3d, pn2_sem))

    cv2.putText(pn, 'PointNet', (20, 100), font,1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pn, 'PointNet', (20, 520), font,1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pn2, 'PointNet2', (20, 100), font,1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(pn2, 'PointNet2', (20, 520), font,1, (255, 255, 255), 2, cv2.LINE_AA)

    merge = np.hstack((pn, pn2))
    reduced_class_names = ['unlabelled', 'vehicle', 'human', 'ground', 'structure', 'nature']
    reduced_colors = [[255, 255, 255],[245, 150, 100],[30, 30, 255],[255, 0, 255],[0, 200, 255],[0, 175, 0]]
    for i,(name,c) in enumerate(zip(reduced_class_names, reduced_colors)):
        cv2.putText(merge, name, (200 + i * 200, 50), font,1, [c[2],c[1],c[0]], 2, cv2.LINE_AA)

    cv2.line(merge,(0,70),(1600,70),(255,255,255),2)
    cv2.line(merge,(800,70),(800,1300),(255,255,255),2)

    merge = cv2.resize(merge,(0,0),fx=0.8,fy=0.8)
    # cv2.imshow('merge', merge)
    # if 32 == waitKey(1):
    #     break
    out.write(merge)

    print(index)
out.release()