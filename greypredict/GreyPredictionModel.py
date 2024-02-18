import numpy as np
import math

class GM11(object):
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.AGO = None
        self.Z = None
        self.a = None
        self.b = None
        self.ori_hat = None

    def model_fit(self):
        self.AGO = self.data.cumsum()
        self.Z = np.array([0.5 * self.AGO[i-1] + 0.5 * self.AGO[i] for i in range(1, len(self.AGO))])
        B = np.mat(np.vstack((-self.Z, np.ones(len(self.Z))))).T
        Y = np.mat(np.delete(self.data, 0)).T
        a_hat = np.linalg.inv(B.T * B) * B.T * Y
        a_hat = np.array(a_hat)
        self.a, self.b = float(a_hat[0]), float(a_hat[1])

    def predict(self, pre_times):
        self.model_fit()
        Times = len(self.data) + pre_times

        def pref(t, a=self.a, b=self.b, x0=self.data[0]):
            return (x0 - b / a) * np.exp(-a * (t - 1)) + b / a

        x1_hat = pref(np.arange(1, Times + 1))
        x0_hat = np.hstack((x1_hat[0], np.diff(x1_hat)))
        x0_pre = x0_hat[len(self.data):]
        ori_hat = x0_hat[:len(self.data)]
        self.ori_hat = ori_hat
        return x0_pre

    def seasonal_fit_and_predict(self, period, pre_times):
        seasonal_operator = []
        total_mean = np.mean(self.data)
        num_periods = math.ceil(len(self.data) / period)
        #print("num period ",num_periods)
        for i in range(period):
            s = 0
            j = 0
            while j < num_periods:
                if i+ period * j < len(self.data): 
                    s += self.data[ i+ period * j]
                else:
                    break
                j += 1
            s /= j
            #print("total mean: ",total_mean)
            seasonal_operator.append(s/total_mean)
        
        seasonal_operator = np.array(seasonal_operator)
        seasonal_transform_data = self.data / np.array([seasonal_operator[i % period] for i in range(len(self.data))])
        self.AGO = seasonal_transform_data.cumsum() 
        self.Z = np.array([0.5 * self.AGO[i-1] + 0.5 * self.AGO[i] for i in range(1, len(self.AGO))])
        B = np.mat(np.vstack((-self.Z, np.ones(len(self.Z))))).T
        Y = np.mat(np.delete(seasonal_transform_data, 0)).T
        a_hat = np.linalg.inv(B.T * B) * B.T * Y
        a_hat = np.array(a_hat)
        self.a, self.b = float(a_hat[0]), float(a_hat[1])

        Times = len(self.data) + pre_times
        def pref(t, a=self.a, b=self.b, x0=seasonal_transform_data[0]):
            return (x0 - b / a) * np.exp(-a * (t - 1)) + b / a

        x1_hat = pref(np.arange(1, Times + 1))
        x0_hat = np.hstack((x1_hat[0], np.diff(x1_hat)* np.array([seasonal_operator[i % period] for i in range(len(np.diff(x1_hat)))] ))) 
        x0_pre = x0_hat[len(self.data):]
        ori_hat = x0_hat[:len(self.data)]
        self.ori_hat = ori_hat
        return x0_pre

class NewInformationGM11(GM11):
    def __init__(self, data):
        super().__init__(data)

    def predict(self, pre_num):
        pre_data = np.ones(pre_num)
        Times = len(self.data)
        for i in range(pre_num):
            self.model_fit()

            def pref(t, a=self.a, b=self.b, x0=self.data[0]):
                return (x0 - b / a) * np.exp(-a * (t - 1)) + b / a

            Times += 1
            x1_hat = pref(np.arange(1, Times + 1))
            x0_hat = np.hstack((x1_hat[0], np.diff(x1_hat)))
            x0_pre = x0_hat[-1]
            self.data = np.append(self.data, x0_pre)
            pre_data[i] = x0_pre
        return pre_data


class MetabolismGM11(GM11):
    def __init__(self, data):
        super().__init__(data)

    def predict(self, pre_num):
        pre_data = np.ones(pre_num)
        Times = len(self.data)
        for i in range(pre_num):
            self.model_fit()

            def pref(t, a=self.a, b=self.b, x0=self.data[0]):
                return (x0 - b / a) * np.exp(-a * (t - 1)) + b / a

            Times += 1
            x1_hat = pref(np.arange(1, Times + 1))
            x0_hat = np.hstack((x1_hat[0], np.diff(x1_hat)))
            x0_pre = x0_hat[-1]
            self.data = np.delete(self.data, 0)
            self.data = np.append(self.data, x0_pre)
            pre_data[i] = x0_pre
        return pre_data
