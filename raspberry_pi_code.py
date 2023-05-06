#import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image
import PIL.Image
import tflite_runtime.interpreter as tflite


IMG_SIZE = (300, 300)
#model = tf.keras.models.load_model("flowers_resent50.h5")
tflite_model = "flower_orginal_tflite_model (2).tflite"
quant_model = "flowers_quantized_keras_int8_model.tflite"
orginal_tflite_model = "flower_orginal_tflite_model (2).tflite"
class_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

"""
# Function to infer the val images
def benchmark(test_data_dir, benchmark_model, classes=class_names, image_size=IMG_SIZE):
    file_count = 0
    infer_times = []
    for (root, dirs, files) in os.walk(test_data_dir):
        for name in files:
            if name.endswith(".jpg"):
                filename = os.path.join(root, name)
                if file_count < 1 :
                    init_timer_start = time.time()
                    img = np.array(Image.open(filename).resize(image_size))
                    pred = benchmark_model.predict(np.expand_dims(img, axis=0))
                    pred_class = class_names[int(np.argmax(pred[0]))]
                    init_timer_end = time.time()
                    init_timer = init_timer_end - init_timer_start
                    file_count += 1
                else:
                    timer_start = time.time()
                    img = np.array(Image.open(filename).resize(image_size))
                    pred = benchmark_model.predict(np.expand_dims(img, axis=0))
                    pred_class = class_names[int(np.argmax(pred[0]))]
                    timer_end = time.time()
                    infer_times.append((timer_end - timer_start))
                    file_count += 1

    return init_timer, np.mean(infer_times), np.std(infer_times), np.std(infer_times, ddof=1)/np.sqrt(np.size(infer_times))
                """
def benchmark_tflite(val_dir, tflite_model, class_names=class_names, image_size=IMG_SIZE):
    interpreter = tflite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    file_count = 0
    infer_times = []
    for (root, dirs, files) in os.walk(val_dir):
        for name in files:
            if name.endswith(".jpg"):
                filename = os.path.join(root, name)
                if file_count < 1 :
                    init_timer_start = time.time()
                    img = (np.array(Image.open(filename).resize(image_size))).astype(np.float32)
                    img = np.expand_dims(img, axis=0)
                    interpreter.set_tensor(input_index, img)
                    interpreter.invoke()
                    output = interpreter.tensor(output_index)
                    pred_class = class_names[int(np.argmax(output()[0]))]
                    init_timer_end = time.time()
                    init_timer = init_timer_end - init_timer_start
                    file_count += 1
                else:
                    timer_start = time.time()
                    img = (np.array(Image.open(filename).resize(image_size))).astype(np.float32)
                    img = np.expand_dims(img, axis=0)
                    interpreter.set_tensor(input_index, img)
                    interpreter.invoke()
                    output = interpreter.tensor(output_index)
                    pred_class = class_names[int(np.argmax(output()[0]))]
                    timer_end = time.time()
                    infer_times.append((timer_end - timer_start))
                    file_count += 1
    return init_timer, np.mean(infer_times), np.std(infer_times), np.std(infer_times, ddof=1)/np.sqrt(np.size(infer_times))
    

def benchmark_tflite_armnn(val_dir, tflite_model, class_names=class_names, image_size=IMG_SIZE):
    armnn_delegate = tflite.load_delegate(library="/home/pi/project/ArmNN-aarch64/libarmnnDelegate.so",
                                       options={"backends": "CpuAcc,CpuRef", "logging-severity":"info"})
    interpreter = tflite.Interpreter(model_path=tflite_model,
                                        experimental_delegates=[armnn_delegate])
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    file_count = 0
    infer_times = []
    for (root, dirs, files) in os.walk(val_dir):
        for name in files:
            if name.endswith(".jpg"):
                filename = os.path.join(root, name)
                if file_count < 1 :
                    init_timer_start = time.time()
                    img = (np.array(Image.open(filename).resize(image_size))).astype(np.float32)
                    img = np.expand_dims(img, axis=0)
                    interpreter.set_tensor(input_index, img)
                    interpreter.invoke()
                    output = interpreter.tensor(output_index)
                    pred_class = class_names[int(np.argmax(output()[0]))]
                    init_timer_end = time.time()
                    init_timer = init_timer_end - init_timer_start
                    file_count += 1
                else:
                    timer_start = time.time()
                    img = (np.array(Image.open(filename).resize(image_size))).astype(np.float32)
                    img = np.expand_dims(img, axis=0)
                    interpreter.set_tensor(input_index, img)
                    interpreter.invoke()
                    output = interpreter.tensor(output_index)
                    pred_class = class_names[int(np.argmax(output()[0]))]
                    timer_end = time.time()
                    infer_times.append((timer_end - timer_start))
                    file_count += 1

    return init_timer, np.mean(infer_times), np.std(infer_times), np.std(infer_times, ddof=1)/np.sqrt(np.size(infer_times))



init_time, avg_time, std, ste = benchmark_tflite(val_dir="flower_photos", tflite_model=tflite_model)
print("TFLITE BENCHMARK\n====================================")
print(f"First image --> {init_time * 1000:.2f} ms")
print(f"Average time --> {avg_time * 1000:.2f} ms")
print(f"Standard deviation --> {std * 1000:.2f} ms")
print(f"Standard error --> {ste * 1000:.2f} ms")
                            
"""
init_time, avg_time, std, ste = benchmark_tflite_armnn(val_dir="flower_photos", tflite_model=tflite_model)
print("TFLITE BENCHMARK\n====================================")
print(f"First image --> {init_time * 1000:.2f} ms")
print(f"Average time --> {avg_time * 1000:.2f} ms")
print(f"Standard deviation --> {std * 1000:.2f} ms")
print(f"Standard error --> {ste * 1000:.2f} ms")

print("\nORGINAL MODEL BENCHMARK\n====================================")

init_time, avg_time, std, ste = benchmark(test_data_dir="flower_photos", benchmark_model=model)
print(f"First image --> {init_time * 1000:.2f} ms")
print(f"Average time --> {avg_time * 1000:.2f} ms")
print(f"Standard deviation --> {std * 1000:.2f} ms")
print(f"Standard error --> {ste * 1000:.2f} ms")
                                            
            """
