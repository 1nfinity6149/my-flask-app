import json
import numpy as np
import geatpy as ea
import requests
import threading
from flask import Flask, request, jsonify

app = Flask(__name__)

def get_params(data) -> dict:
    # 从请求数据中获取参数，如果不存在则使用默认值
    return {
        "model_params": {
            "A": data.get('A', 1.9),
            "m": data.get('m', 0.4),
            "L": data.get('L', [1300, 1200, 1250, 1260, 1300]),
            "Q_design": data.get('Q_design', [0.2, 0.2, 0.3, 0.5, 0.2]),
            "Q_sp": data.get('Q_sp', 5),
            "t_max": data.get('t_max', 100),
            "lz": data.get('lz', [870, 1150, 1740, 2410, 3100]),
            "area": data.get('area', [780, 920, 740, 1000, 680]),
            "lb_r": data.get('lb_r', 0.6),
            "ub_r": data.get('ub_r', 1.2),
            "waterPerMu": data.get('waterPerMu', 20)
        },
        "ga_params": {
            "pop": {
                "Encoding": "RI",
                "NIND": data.get('NIND', 50)
            },
            "other": {
                "MAXGEN": data.get('MAXGEN', 200),
                "logTras": 0
            }
        }
    }

class HHWProblem(ea.Problem):
    def __init__(self, params):
        self.A = params['A']
        self.m = params['m']
        self.L = params['L']
        self.Q_design = np.array(params['Q_design'])[None]
        self.Q_sp = params['Q_sp']
        self.t_max = params['t_max']
        self.lz = np.array(params['lz'])
        self.area = np.array(params['area'])
        self.lb_r = params['lb_r']
        self.ub_r = params['ub_r']
        self.waterPerMu = params['waterPerMu']
        self.numberOfChannels = len(self.L)

        self.W = self.area * self.waterPerMu

        args = self.init_params()
        super(HHWProblem, self).__init__(*args)

    def init_params(self):
        name = 'HHRiver'
        M = 1
        maxormins = np.array([1])
        Dim = int(self.numberOfChannels * 2)
        varTypes = np.array([0 for i in range(Dim)])

        lb = np.concatenate((np.zeros(self.numberOfChannels), self.lb_r * np.ones(self.numberOfChannels)))
        ub = np.concatenate((self.t_max * np.ones(self.numberOfChannels), self.ub_r * np.ones(self.numberOfChannels)))

        return [name, M, maxormins, Dim, varTypes, lb, ub]

    def calculateQAdapted(self, alpha):
        Q_adapted = self.Q_design * alpha
        return np.tile(Q_adapted[:, None, :], (1, self.t_max + 1, 1))

    def seepageCalculation(self, Q, L):
        return (self.A * L * (Q ** (self.m - 1)) * 3600) / 100

    def evalVars(self, Vars):
        B = Vars.shape[0]
        startTime = Vars[:, :self.numberOfChannels]
        alpha = Vars[:, self.numberOfChannels:]
        Q_adapted = self.calculateQAdapted(alpha)

        ObjV = np.zeros([B, 1])
        CV = np.zeros([B, 1])

        openStatus = np.zeros((B, self.t_max + 1, self.numberOfChannels))
        for i in range(self.numberOfChannels):
            for b in range(B):
                endTime = np.minimum(self.t_max, startTime[b, i] + self.W[i] / Q_adapted[b, np.round(startTime[b, i]).astype(np.int32), i])
                openStatus[b, np.round(startTime[b, i]).astype(np.int32):np.round(endTime).astype(np.int32) + 2, i] = 1

        Q_total_per_time_step = np.sum(Q_adapted * openStatus, axis=2)

        for t in range(self.t_max + 1):
            for b in range(B):
                activeChannels = np.where(openStatus[b, t] > 0)[0]
                if len(activeChannels) > 0:
                    L_first = self.lz[activeChannels[0]]
                    ObjV[b] += self.seepageCalculation(Q_adapted[b, t, activeChannels[0]] + 1e-4, L_first)

                    for i in range(1, len(activeChannels)):
                        L_segment = self.lz[activeChannels[i]] - self.lz[activeChannels[i - 1]]
                        Q_segment = Q_total_per_time_step[b, t] - np.sum(Q_adapted[b, t, activeChannels[0:i]])
                        ObjV[b] += self.seepageCalculation(Q_segment + 1e-4, L_segment)

        CV = Q_total_per_time_step - self.Q_sp
        return ObjV, CV

    def forward(self, Vars):
        bestSolution = Vars.squeeze(axis=0)
        startTime = bestSolution[:self.numberOfChannels]
        alpha = bestSolution[self.numberOfChannels:]
        Q_adapted = self.calculateQAdapted(alpha)
        Q_adapted = Q_adapted.squeeze()

        openStatus = np.zeros((self.t_max + 1, self.numberOfChannels))
        cyclesTimes = np.zeros(self.numberOfChannels)
        lossWater = 0

        for i in range(self.numberOfChannels):
            endTime = np.minimum(self.t_max, startTime[i] + self.W[i] / Q_adapted[np.round(startTime[i]).astype(np.int32), i])
            openStatus[np.round(startTime[i]).astype(np.int32):np.round(endTime).astype(np.int32) + 2, i] = 1
            cyclesTimes[i] = 1  # 每个闸门打开一次

            # 计算每个闸门的输水损失
            for t in range(int(np.round(startTime[i])), int(np.round(endTime)) + 1):
                lossWater += self.seepageCalculation(Q_adapted[t, i], self.lz[i])

        Q_ad = Q_adapted * openStatus
        return Q_ad, startTime, lossWater, cyclesTimes

def calculate_model(data):
    try:
        params = get_params(data)
        problem = HHWProblem(params['model_params'])
        algorithm = ea.soea_SEGA_templet(
            problem,
            ea.Population(**params['ga_params']['pop']),
            **params['ga_params']['other']
        )
        algorithm.recOper.XOVR = 0.9
        algorithm.mutOper.Pm = 0.01
        res = ea.optimize(
            algorithm,
            seed=1,
            verbose=False,
            drawing=0,
            outputMsg=False,
            drawLog=False,
            saveFlag=False
        )

        if res['success']:
            Q_ad, startTime, lossWater, cyclesTimes = problem.forward(res['Vars'])
            result_data = {
                "name": "2024年冬灌第二轮配水方案-20241017-005",
                "req": {
                    "totalDemandWater": "100",
                    "totalAllocatedWater": "80",
                    "irrigationArea": "100",
                    "shortageRate": "0",
                    "demandSatisfactionRate": "0.8",
                    "schedulingTarget": "1",
                    "irrigationYear": "2024",
                    "irrigationSeason": "3",
                    "irrigationCycle": "第二轮",
                    "irrigationStartTime": "2024-02-01",
                    "irrigationEndTime": "2024-02-10"
                },
                "schedulingScheme": [
                    {
                        "nodeId": "497",
                        "nodeName": "新城支管_8#分水口",
                        "controllerIDRPName": "单元497",
                        "status": 2,
                        "remarks": None,
                        "join": True,
                        "flow": "1.000"
                    },
                    {
                        "nodeId": "498",
                        "nodeName": "六公里支管_1#分水口",
                        "controllerIDRPName": "单元498",
                        "status": 2,
                        "remarks": None,
                        "join": True,
                        "flow": "1.000"
                    }
                ],
                "nodeJoinNum": "8",
                "operatingCosts": "100",
                "lossWater": str(lossWater),
                "cyclesTimes": str(np.sum(cyclesTimes)),
                "schedulingDuration": "10",
                "timeSpan": startTime.tolist(),
                "supplyVolume": Q_ad.tolist()
            }
            # 打印发送的数据
            print("发送的数据:", json.dumps(result_data, indent=4, ensure_ascii=False))

            # 发送结果到数据库
            response = requests.post(
                "http://171.15.113.252:20010/admin-api/schedule/schemeAfterCall",
                json=result_data
            )
            if response.status_code == 200:
                print(f"数据发送成功: {response.status_code}")
            else:
                print(f"数据发送失败，状态码: {response.status_code}, 响应内容: {response.text}")
        else:
            print("优化未成功。")
            print(f"优化结果: {res}")
    except Exception as e:
        print(f"发生异常: {e}")
        # 在异常发生时发送错误信息到数据库
        error_data = {
            'error': str(e)
        }
        try:
            requests.post("http://171.15.113.252:20010/admin-api/schedule/schemeAfterCall", json=error_data)
        except Exception as inner_e:
            print(f"无法发送错误信息到数据库: {inner_e}")

@app.route('/run_model', methods=['POST'])
def run_model():
    data = request.json
    print("接收到的请求数据:", data)

    threading.Thread(target=calculate_model, args=(data,)).start()
    return jsonify({"message": "模型计算预计十分钟内完成！"}), 202

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
