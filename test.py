# from keras.models import load_model
# import numpy as np
# import copy
# import itertools


# class Test:
#     def __init__(self):
#         self.test_model()


#     def test_model(self):
#         savedModel=load_model('models/keypoint_classifier.hdf5')


#         temp_landmark_list = self.pre_process_landmark()

#         print('temp_landmark_list', temp_landmark_list, '\n\n')




#         temp_landmark_list = np.expand_dims(t, axis=0)
#         prediction = savedModel.predict(temp_landmark_list)
#         print(prediction)
#         prediction_index = np.argmax(prediction)
#         print("Index of highest value in prediction:", prediction_index)


#     def pre_process_landmark(self, landmark_list):
#         temp_landmark_list = copy.deepcopy(landmark_list)

#         print('temp_landmark_list-1', temp_landmark_list, '\n')


#         # 相対座標に変換
#         base_x, base_y = 0, 0
#         for index, landmark_point in enumerate(temp_landmark_list):
#             if index == 0:
#                 base_x, base_y = landmark_point[0], landmark_point[1]

#             temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
#             temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

#         # 1次元リストに変換

#         print('temp_landmark_list0', temp_landmark_list, '\n')

#         temp_landmark_list = list(
#             itertools.chain.from_iterable(temp_landmark_list))
        
#         print('temp_landmark_list1', temp_landmark_list, '\n')

#         # 正規化
#         max_value = max(list(map(abs, temp_landmark_list)))

#         def normalize_(n):
#             return n / max_value
        

#         temp_landmark_list = list(map(normalize_, temp_landmark_list))

#         print('temp_landmark_list3', temp_landmark_list, '\n')

#         return temp_landmark_list
            





