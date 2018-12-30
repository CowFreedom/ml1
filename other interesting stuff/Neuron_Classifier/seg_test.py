try:
    from PIL import Image
    from PIL import ImageDraw
    import numpy as np
except ImportError:
    import Image
    import ImageDraw
    print("IMPORT ERROR")

import numpy as np
import segmentator as sg


def testImage():
	img=Image.open('D://Documents//Uni//Programming//Machine Learning Tutorium//github Ordner//other interesting stuff//Neuron_Classifier//data//image2.jpg')
	#print(Image.open('D://Documents//Uni//Programming//Machine Learning Tutorium//github Ordner//other interesting stuff//Neuron_Classifier//data//image1.jpg').size)
			
	segmented_image=sg.segmentator(np.asarray(img))
	
testImage()