# -*- coding: utf-8 -*-
"""
Controlador Final de Navegação Inteligente (CNN + MLP + RB)

Autor: Você
"""

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from controller import Robot
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

MODEL_PATH = "robot_perception_model.pth"
MAX_SPEED = 4.0
IMG_HEIGHT = 64
IMG_WIDTH = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HybridNavNet(nn.Module):
    def __init__(self, lidar_input_size):
        super(HybridNavNet, self).__init__()
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.lidar_branch = nn.Sequential(
            nn.Linear(lidar_input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.regressor_head = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 64, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, image, lidar):
        cnn_out = self.cnn_branch(image)
        lidar_out = self.lidar_branch(lidar)
        return self.regressor_head(torch.cat((cnn_out, lidar_out), dim=1))

def build_bayesian_network():
    bn = DiscreteBayesianNetwork([
        ('TargetVisible', 'Direction'),
        ('ObstacleDetected', 'Direction'),
        ('Direction', 'Action')
    ])
    cpd_target = TabularCPD('TargetVisible', 2, [[0.5], [0.5]])
    cpd_obstacle = TabularCPD('ObstacleDetected', 2, [[0.5], [0.5]])
    cpd_direction = TabularCPD(
        'Direction', 3,
        [
            [0.45, 0.5, 0.01, 0.35],  # esquerda
            [0.1,  0.0, 0.98, 0.3],  # frente
            [0.45, 0.5, 0.01, 0.35]   # direita
        ],
        evidence=['TargetVisible', 'ObstacleDetected'],
        evidence_card=[2, 2]
    )
    cpd_action = TabularCPD(
        'Action', 3,
        [
            [0.05, 0.96, 0.05],  # seguir
            [0.92,  0.02, 0.03], # virar esquerda
            [0.03, 0.02, 0.92]   # virar direita
        ],
        evidence=['Direction'],
        evidence_card=[3]
    )
    bn.add_cpds(cpd_target, cpd_obstacle, cpd_direction, cpd_action)
    assert bn.check_model()
    return VariableElimination(bn)

def map_to_probabilities(dist_pred, angle_pred):
    prob_obstacle = 1 / (1 + np.exp(12 * (dist_pred - 0.5)))
    prob_target = np.exp(-2.5 * (angle_pred**2))
    return np.clip(prob_target, 0.01, 0.99), np.clip(prob_obstacle, 0.01, 0.99)

def preprocess_input(camera, image_data, lidar_data):
    image = np.frombuffer(image_data, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.transpose((2, 0, 1)) / 255.0
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)
    lidar_data[lidar_data == np.inf] = 5.0
    lidar_tensor = torch.from_numpy(lidar_data).float().unsqueeze(0)
    return image_tensor.to(device), lidar_tensor.to(device)

def set_motor_speeds(motors, left, right):
    motors["front_left"].setVelocity(left)
    motors["back_left"].setVelocity(left)
    motors["front_right"].setVelocity(right)
    motors["back_right"].setVelocity(right)

def main():
    robot = Robot()
    TIME_STEP = int(robot.getBasicTimeStep())
    camera = robot.getDevice("camera")
    camera.enable(TIME_STEP)
    lidar = robot.getDevice("Sick LMS 291")
    lidar.enable(TIME_STEP)

    motors = {
        "front_left": robot.getDevice("front left wheel"),
        "front_right": robot.getDevice("front right wheel"),
        "back_left": robot.getDevice("back left wheel"),
        "back_right": robot.getDevice("back right wheel"),
    }
    for motor in motors.values():
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)

    model = HybridNavNet(lidar.getHorizontalResolution()).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Modelo carregado!")

    bn_inference = build_bayesian_network()

    while robot.step(TIME_STEP) != -1:
        image_data = camera.getImage()
        if not image_data:
            continue
        lidar_data = np.array(lidar.getRangeImage())
        image_tensor, lidar_tensor = preprocess_input(camera, image_data, lidar_data)

        with torch.no_grad():
            dist_pred, angle_pred = model(image_tensor, lidar_tensor)[0].cpu().numpy()

        
        prob_target, prob_obstacle = map_to_probabilities(dist_pred, angle_pred)

        img_for_cv = np.frombuffer(image_data, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        hsv = cv2.cvtColor(cv2.cvtColor(img_for_cv, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        is_yellow_visible = np.sum(mask) > 800
        if is_yellow_visible:
            prob_target = 0.97

        if is_yellow_visible and dist_pred < 0.25:
            print("Alvo alcançado. Parando!")
            set_motor_speeds(motors, 0, 0)
            break

        # --- NOVO REFLEXO MAIS SIMPLES E ROBUSTO ---
        
        if dist_pred < 0.30:
            # Analisa qual lado tem mais espaço livre
            num_rays = lidar.getHorizontalResolution()
            max_range = lidar.getMaxRange()
            ranges = np.array(lidar_data)
            ranges[ranges == np.inf] = max_range
            normalized = 1.0 - (ranges / max_range)
        
            # Calcula "quanto espaço livre" tem em cada lado com peso Gaussian
            left_weights = np.exp(-((np.arange(num_rays) - num_rays * 0.2)**2) / (2 * (num_rays * 0.1)**2))
            right_weights = np.exp(-((np.arange(num_rays) - num_rays * 0.7)**2) / (2 * (num_rays * 0.1)**2))
        
            left = np.sum(normalized * left_weights)
            right = np.sum(normalized * right_weights)
        
            # Se tiver mais espaço na esquerda, sempre vira esquerda, caso contrário direita
            if left <= right:
                set_motor_speeds(motors, -0.5 * MAX_SPEED, 0.5 * MAX_SPEED)
                print(f"REFLEXO: dist={dist_pred:.2f} | Vira ESQUERDA (L={left:.2f} R={right:.2f})")
            else:
                set_motor_speeds(motors, 0.5 * MAX_SPEED, -0.5 * MAX_SPEED)
                print(f"REFLEXO: dist={dist_pred:.2f} | Vira DIREITA (L={left:.2f} R={right:.2f})")
        
            continue

        evidence = {
            'TargetVisible': int(prob_target >= 0.7),
            'ObstacleDetected': int(prob_obstacle >= 0.5)
        }

        cpd_target = TabularCPD('TargetVisible', 2, [[1 - prob_target], [prob_target]])
        cpd_obstacle = TabularCPD('ObstacleDetected', 2, [[1 - prob_obstacle], [prob_obstacle]])

        query = bn_inference.query(
            variables=['Action'],
            evidence=evidence,
            virtual_evidence=[cpd_target, cpd_obstacle]
        )

        action = np.argmax(query.values)
        actions = ['SEGUIR', 'VIRAR ESQUERDA', 'VIRAR DIREITA']
        print(f"Dist: {dist_pred:.2f}m | Angle: {np.degrees(angle_pred):.1f}° | "
              f"P(T): {prob_target:.2f} | P(O): {prob_obstacle:.2f} | Action: {actions[action]}")

        if action == 0:
            set_motor_speeds(motors, MAX_SPEED, MAX_SPEED)
        elif action == 1:
            set_motor_speeds(motors, -0.5 * MAX_SPEED, 0.5 * MAX_SPEED)
        elif action == 2:
            set_motor_speeds(motors, 0.5 * MAX_SPEED, -0.5 * MAX_SPEED)

if __name__ == "__main__":
    main()
