import json
import numpy as np
import geatpy as ea
from flask import Flask, request, jsonify
import threading
import requests  # 导入requests库以发送HTTP请求

app = Flask(__name__)
def get_params(json_file: str='myconfig.json') -> dict:
    with open(json_file, 'r', encoding='utf-8') as load_f:
        params = json.load(load_f)
    return params
class HHWProblem(ea.Problem):
    def __init__(self, params):
        self.n_crops = params['n_crops']
        self.reservoir_capacity = params['reservoirs']['reservoir_capacity']
        self.n_reservoir_plots = params['reservoirs']['n_reservoir_plots']
        self.n_plots = sum(self.n_reservoir_plots)
        self.water_demand = np.array(params['water_demand'])
        self.weights = params['weights']
        self.water_efficiency = params['water_efficiency']
        self.economic_benefit = np.array(params['economic_benefit'])
        if params['user_params']['is_input']:
            self.user_percentage = np.array(params['user_params']['user_percentage'])
        self.n_reservoir = len(self.reservoir_capacity)

        args = self.init_params()
        super(HHWProblem, self).__init__(*args)

    def init_params(self):
        name = 'HHRiver'
        M = 1
        maxormins = np.array([1])
        Dim = int(self.n_plots * self.n_crops)
        varTypes = np.array([0 for i in range(Dim)])
        lb = np.zeros([Dim])
        ub = []

        for n in range(self.n_reservoir):
            for i in range(self.n_reservoir_plots[n]):
                ub.append(
                    self.reservoir_capacity[n] / self.n_reservoir_plots[n]
                )
        ub = np.array(ub)
        return [name, M, maxormins, Dim, varTypes, lb, ub]
    def evalVars(self, Vars):
        water_allocation = np.reshape(Vars, [-1, self.n_plots, self.n_crops])
        water_shortage = np.maximum(0, self.water_demand[None] - water_allocation * self.water_efficiency)
        total_shortage = np.sum(water_shortage, axis=(1, 2), keepdims=True)
        total_benefit = np.sum(water_allocation * self.economic_benefit[None, None], axis=(1, 2), keepdims=True)
        ObjV = self.weights[0] * total_shortage - self.weights[1] * total_benefit
        ObjV = np.squeeze(ObjV, axis=-1)
        # % 分别计算每个水库的配水量，并考虑输水效率
        CV = []
        start_index = 0
        for i in range(self.n_reservoir):
            end_index = start_index + self.n_reservoir_plots[i]
            # 非线性不等式约束：水量不能超过可用水资源
            CV.append(
                np.sum(water_allocation[:, start_index:end_index, :],axis=(1, 2)) * self.water_efficiency - self.reservoir_capacity[i]
            )
            start_index = end_index
        # % 非线性不等式约束：水量不能超过可用水资源
        CV = np.stack(CV, axis=-1)
        return ObjV, CV
    def forward(self, Vars):
        best_water_allocation = np.reshape(Vars.squeeze(axis=0), [self.n_plots, self.n_crops])
        total_water_per_plot = np.sum(best_water_allocation, 1) * self.water_efficiency
        total_water_demand_per_plot = np.sum(self.water_demand, 1)
        water_shortage_per_plot = total_water_demand_per_plot - total_water_per_plot
        return total_water_per_plot, water_shortage_per_plot

# 模型计算任务
def async_model_calculation(input_data):
    params = get_params()
    problem = HHWProblem(params['model_params'])

    algorithm = ea.soea_SEGA_templet(problem, ea.Population(**params['ga_params']['pop']),
                                     **params['ga_params']['other'])
    res = ea.optimize(algorithm, seed=1, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
    algorithm.problem = problem
    if res['success']:
        total_water_per_plot, water_shortage_per_plot = problem.forward(res['Vars'])

        # 更新输入数据中的每个水源的配水量和缺水量
        for i, water_source in enumerate(input_data['waterSource']):
            water_source['configVolume'] = round(total_water_per_plot[i], 4)  # 配水量，保留小数点后4位
            water_source['configshortageVolume'] = round(water_shortage_per_plot[i], 4)  # 缺水量，保留小数点后4位

        # 将更新后的数据发送到目标URL
        target_url = 'http://171.15.113.252:20010/admin-api/schedule/configureAfterCall'
        try:
            response = requests.post(target_url, json=input_data)
            if response.status_code == 200:
                print("更新的数据成功发送到目标URL。")
                print(f"发送的数据: {json.dumps(input_data, indent=4)}")  # 打印发送的数据
            else:
                print(f"发送失败，状态码: {response.status_code}, 响应: {response.text}")
        except Exception as e:
            print(f"请求失败: {e}")

# 第一个接口：接收请求并启动异步计算任务
@app.route('/api/request_model', methods=['POST'])
def request_model():
    data = request.json

    # 启动一个新的线程，进行异步计算
    threading.Thread(target=async_model_calculation, args=(data,)).start()

    return jsonify({"message": "模型计算预计十分钟内完成！"}), 202

# 第二个接口：接收入参并进行约束检查和返回结果
@app.route('/api/check_constraints', methods=['POST'])
def check_constraints():
    data = request.json

    # 约束检查 1：检查所有水库的配置比例是否超过 100%
    for water_source in data['waterSource']:
        config_ratio = float(water_source.get('configRate', 0))
        if config_ratio > 1:
            return jsonify({"code": 400, "msg": "配置比例超出100%，请调整"}), 400

    # 约束检查 2：检查灌片1、灌片2、灌片3的总配置量是否超过水库可供水量 50 万
    total_config_volume = sum(
        float(water_source.get('configVolume', 0))
        for water_source in data['waterSource']
        if water_source['irrigationId'] in ['1', '2', '3']
    )

    if total_config_volume > 50:
        return jsonify({"code": 403, "msg": "灌片1+灌片2+灌片3总配置量超过水源水库可供水量50万，需调整"}), 403

    # 更新配水量和缺水量
    for water_source in data['waterSource']:
        config_ratio = float(water_source.get('configRate', 0))
        water_source['configVolume'] = float(water_source['supplyVolume']) * config_ratio
        water_source['configshortageVolume'] = float(water_source['forecastDemandVolume']) - water_source['configVolume']

    # 成功响应
    return jsonify({"code": 200, "msg": "配置成功"}), 200



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=20000, debug=True)
