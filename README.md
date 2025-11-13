# 🤖 Autonomous Robot Navigation – Webots Simulation  
**(CNN + MLP + Rede Bayesiana + Reflexo Heurístico Integrado)**

Simulação de navegação autônoma no **Webots** que combina percepção profunda (CNN + MLP), inferência probabilística (Rede Bayesiana) e um **reflexo heurístico de emergência** para evitar colisões. Projeto acadêmico/educacional para comparação e validação de estratégias híbridas de controle.

---

## 📁 Estrutura do Repositório

```
autonomous-robot-webots/
├── worlds/
│   └── IA-20251.wbt
│
├── controllers/
│   ├── intelligent_navigator_controller/
│   │   ├── intelligent_navigator_controller.py   # Controlador híbrido (CNN+MLP+RB + reflexo)
│   │   └── robot_perception_model.pth            # Pesos PyTorch do modelo de percepção
│   │
│   └── navigator_controller/
│       ├── navigator_controller.py              
│       └── robot_perception_model.pth            
│
├── requirements.txt
└── README.md
```

---

## 🔬 Objetivo

Construir e testar um controlador híbrido que:
- Use **visão (câmera)** e **LIDAR** para percepção;
- Preveja **distância** e **ângulo** ao alvo via uma rede híbrida (CNN + MLP);
- Converta essas estimativas em probabilidades e realize **inferência** numa **Rede Bayesiana** para escolher a ação;
- **Atue reflexivamente** (regra de emergência) quando um obstáculo estiver muito próximo, priorizando segurança.

---

## 🧠 Descrição técnica do controlador (`intelligent_navigator_controller.py`)

### Principais blocos:
1. **Percepção híbrida (HybridNavNet)**  
   - CNN branch para imagens (3 conv layers + pooling).  
   - LIDAR branch (MLP).  
   - Head regressora que prediz `[dist_pred, angle_pred]`.

2. **Mapeamento para probabilidades**  
   - `map_to_probabilities(dist_pred, angle_pred)` converte as saídas contínuas em:
     - `prob_obstacle` (probabilidade de obstáculo perigoso)
     - `prob_target` (probabilidade de objetivo visível / relevante)

3. **Verificação visual direta do alvo**  
   - Segmentação HSV para detectar cor amarela (alvo).  
   - Se alvo visível e perto → encerra com sucesso.

4. **Reflexo heurístico de emergência** (NOVO)  
   - Ativado quando `dist_pred < 0.30` m (threshold configurável).  
   - Calcula qual lado (esquerdo/direito) tem **mais espaço livre** usando leitura do LIDAR e janelas gaussianas de ponderação.  
   - Executa uma manobra imediata (giro esquerdo/direito) e **continua** o loop (não consulta a RB nesse passo).

5. **Rede Bayesiana (pgmpy)**  
   - Variáveis: `TargetVisible`, `ObstacleDetected`, `Direction`, `Action`.  
   - Evidência virtual (TabularCPD) atualizada a cada step.  
   - Inferência via `VariableElimination` para escolher `Action` ∈ {SEGUIR, VIRAR_ESQ, VIRAR_DIR}.

6. **Atuação**  
   - Mapeia `Action` para velocidades dos motores (`set_motor_speeds`), com `MAX_SPEED = 4.0`.

---

## ⚙️ Requisitos (ambiente)

- **Webots** (R2023b ou superior recomendado)  
- **Python 3.8+** executado pelo Webots controller (ou ambiente que Webots usa)
- Bibliotecas Python:

```bash
pip install torch numpy opencv-python pgmpy
```

> 💡 *Em ambientes sem GPU, o PyTorch instala automaticamente a versão CPU.*

---

## ▶️ Como executar

1. Abra o **Webots**.  
2. Carregue o mundo:
   ```text
   File → Open World → worlds/IA-20251.wbt
   ```
3. No nó do robô, selecione o controller:
   ```text
   intelligent_navigator_controller
   ```
   - Certifique-se de que o arquivo `intelligent_navigator_controller/robot_perception_model.pth` esteja presente.  
4. Pressione ▶️ **Play** para iniciar a simulação.  
5. Observe o console do Webots — mensagens como:
   ```text
   Modelo carregado!
   Dist: 0.42m | Angle: 12.5° | P(T): 0.83 | P(O): 0.31 | Action: SEGUIR
   REFLEXO: dist=0.25 | Vira ESQUERDA (L=1.23 R=0.67)
   ```

---

## 🔧 Parâmetros importantes (onde ajustar)

- `MODEL_PATH` — caminho para pesos PyTorch (`robot_perception_model.pth`)  
- `IMG_HEIGHT`, `IMG_WIDTH` — dimensão de entrada da CNN (64 × 64 por padrão)  
- `MAX_SPEED` — velocidade máxima dos motores  
- **Threshold do reflexo:** `dist_pred < 0.30` (m) — ajusta a sensibilidade de segurança  
- **Pesos gaussianos:** configuram a sensibilidade lateral no reflexo (`num_rays`, `σ`)  

---

## 🧪 Comportamento esperado & testes

- **Cenários típicos**:
  - Alvo visível → confiança alta em `TargetVisible` → ação `SEGUIR`/`APPROACH`.  
  - Obstáculo próximo → reflexo de emergência gira para o lado com mais espaço.  
  - Situações incertas → RB combina evidências e escolhe ação probabilisticamente.  

- **Métricas úteis para avaliação**:
  - Tempo até alcançar o alvo (quando consegue).  
  - Número de intervenções reflexas (quantas vezes reflexo foi acionado).  
  - Colisões ou contatos com obstáculos.  
  - Distância mínima ao obstáculo durante a navegação.  

---

## ☑️ Boas práticas e limitações

- Arquive **pesos grandes** (datasets ou checkpoints extensos) fora do repositório — use releases GitHub, Artifactory ou links para armazenamento (Google Drive, S3).  
- A inferência em tempo real pode exigir CPU moderado; se usar GPU, configure Webots para usar intérprete com CUDA se disponível.  
- A lógica de reflexo é propositalmente simples e segura — pode ser aprimorada por heurísticas mais finas ou por uma política de controle aprendido.  

---

## 👥 Autoria / Colaboradores

- **Dely Silva** — desenvolvimento do controlador híbrido e integração  

*(adicione outros colaboradores conforme necessário)*

---

## 📂 Arquivos chave

- `controllers/intelligent_navigator_controller/intelligent_navigator_controller.py` — controlador completo (percepção, RB, reflexo).  
- `controllers/intelligent_navigator_controller/robot_perception_model.pth` — pesos do modelo PyTorch (não incluídos automaticamente; coloque aqui).  
- `worlds/IA-20251.wbt` — mundo Webots de teste.  

---

## 🔜 Próximos passos sugeridos

- Coletar métricas automáticas e salvar resultados (`.csv`) por execução.  
- Implementar fallback para recarregar o modelo se `robot_perception_model.pth` não for encontrado.  
- Ajustar thresholds e ponderações do reflexo via parâmetros externos (`config.json`).  
- Documentar procedimento de treinamento do `robot_perception_model.pth` em `docs/` (se desejar incluir instruções de re-treino).  

---

> ✅ Este README descreve o controlador **híbrido** (CNN + MLP + Rede Bayesiana) com o reflexo heurístico integrado — pronto para uso em simulações Webots e comparações experimentais.
