import matplotlib.pyplot as plt

# x axis: scale factor
# y axis: IOC/Number of detections
X = [1.1,1.2,1.3,1.4,1.5]
 
# Y11 = [0.011, 0.013, 0.015, 0.015, 0.018]
# Y22 = [0.011, 0.013, 0.020, 0.020, 0.022]
# Y33 = [0.012, 0.012, 0.019, 0.019, 0.024]
# Y44 = [0.012, 0.013, 0.020, 0.016, 0.020]

# Y11 = [10.14, 6.70, 4.68, 3.87, 4.28]
# Y22 = [8.44, 5.25, 3.41, 2.73, 3.02]
# Y33 = [7.15, 4.46, 2.70, 1.89, 2.14]
# Y44 = [6.26, 3.87, 2.02, 1.46, 1.66]

# Y11 = [0.013, 0.011, 0.006, 0.001, 0.001]
# Y22 = [0.014, 0.008, 0.003, 0.0, 0.0]
# Y33 = [0.014, 0.004, 0.0, 0.0, 0.0]
# Y44 = [0.014, 0.001, 0.0, 0.0, 0.0]

Y11 = [0.34, 0.24, 0.14, 0.18, 0.1]
Y22 = [0.28, 0.2, 0.12, 0.12, 0.06]
Y33 = [0.28, 0.18, 0.08, 0.08, 0.06]
Y44 = [0.26, 0.14, 0.04, 0.06, 0.06]

plt.plot(X, Y11, label = "Min neighbors 3")
plt.plot(X, Y22, label = "Min neighbors 4")
plt.plot(X, Y33, label = "Min neighbors 5")
plt.plot(X, Y44, label = "Min neighbors 6")

plt.title('Number of detections for otsedom cascade')
plt.xlabel('Scale factor')
plt.xticks(X)
plt.ylabel('Average number of detections')

plt.legend()
plt.show()