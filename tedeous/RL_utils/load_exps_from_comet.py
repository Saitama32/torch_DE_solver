from comet_ml import API
import torch
import io, os
from datetime import datetime
from tedeous.rl_algorithms import PrioritizedReplayBuffer
from tedeous.RL_utils.load_transitions_into_buffer_pickle import load_transitions_to_replay_buffer


# === Настройки ===
WORKSPACE = "saitama32"
PROJECT_NAME = "rlpinn-diffusion-1d-farm-transitions"
# MAX_EXPERIMENTS = 15  # можно изменить при необходимости

api = API(api_key="aP71fQTYPNqfsYWvudPPmoBl5")  # или просто API()


# === Вспомогательные функции ===
def get_metadata_field(exp, field, default=None):
    try:
        meta = exp.get_metadata()
        return meta.get(field, default)
    except Exception:
        return default
    

def get_param_value(exp, param_name, default=None):
    try:
        params = exp.get_parameters_summary()
        params_dict = {p["name"]: p["valueCurrent"] for p in params}
        return params_dict.get(param_name, default)
    except Exception:
        return default



def get_end_time(exp):
    end_ms = get_metadata_field(exp, "endTimeMillis")
    if end_ms:
        return datetime.fromtimestamp(end_ms / 1000)
    return datetime.min


def get_duration_hours(exp):
    """Возвращает длительность эксперимента в часах."""
    start_ms = get_metadata_field(exp, "startTimeMillis", 0)
    end_ms = get_metadata_field(exp, "endTimeMillis", 0)
    if not start_ms or not end_ms:
        return 0.0
    duration_h = (end_ms - start_ms) / (1000 * 60 * 60)
    return duration_h


def is_crashed(exp):
    return get_metadata_field(exp, "hasCrashed", False) is True


# === Основная функция ===
def collect_all_comet_transitions(replay_buffer=None, max_exps_last=10, duration_grater_hours = 1, save_dir=None, tolerance = 0.0, prev_tol=0.0) -> PrioritizedReplayBuffer:
    """Собирает все переходы из не-crashed экспериментов проекта и возвращает заполненный PrioritizedReplayBuffer."""
    print("🔍 Получаем эксперименты из Comet...")
    experiments = list(api.get_experiments(workspace=WORKSPACE, project_name=PROJECT_NAME))
    # valid_experiments = [exp for exp in experiments if not is_crashed(exp)]
    experiments_sorted = sorted(experiments, key=get_end_time, reverse=True)
    experiments_sorted_duration = [
        exp for exp in experiments_sorted
        if get_duration_hours(exp) >= duration_grater_hours
    ]
    # experiments_sorted = [api.get_experiment(workspace=WORKSPACE, project_name=PROJECT_NAME, experiment='751c7ca595dd4dafb22a0cfe61c26b6f')]

    experiments_sorted_duration = experiments_sorted_duration[:max_exps_last]
    if prev_tol>0.0:

        experiments_sorted_tol = [
            exp for exp in experiments_sorted_duration 
            if float(get_param_value(exp, "tolerance", 0.0)) >= prev_tol
        ]
    else:
        experiments_sorted_tol = [
            exp for exp in experiments_sorted_duration 
            if float(get_param_value(exp, "tolerance", 0.0)) >= tolerance
        ]

    print(f"✅ Найдено {len(experiments_sorted_tol)} активных экспериментов для загрузки буферов.\n")

    all_transitions = []  # сюда соберём всё

    for i, exp in enumerate(experiments_sorted_tol, 1):
        meta = exp.get_metadata()
        exp_id = meta.get("experimentKey")
        exp_name = meta.get("experimentName")
        print(f"[{i:2d}] {exp_name} ({exp_id})")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            print(f"💾 Сохранение включено — файлы будут сохраняться в {save_dir}")

        assets = exp.get_asset_list()
        # --- фильтруем и сортируем по step ---
        pt_assets = [a for a in assets if a["fileName"].endswith(".pt") and "entry_step" in a["fileName"]]

        def get_step(asset):
            if "step" in asset and isinstance(asset["step"], (int, float)):
                return int(asset["step"])
            fname = asset.get("fileName", "")
            try:
                return int(fname.split("entry_step_")[-1].split(".")[0])
            except Exception:
                return 0

        pt_assets = sorted(pt_assets, key=get_step)

        if not pt_assets:
            print("   ⚠️ Нет файлов entry_step_*.pt — пропускаем.")
            continue

        for asset in pt_assets:
            filename = asset["fileName"]

            try:
                file_bytes = exp.get_asset(asset["assetId"], return_type="binary")
                buffer_stream = io.BytesIO(file_bytes)
                data_load = torch.load(buffer_stream, map_location="cpu")

                if save_dir is not None:
                    safe_name = f"{exp_name}_{filename}".replace("/", "_")
                    save_path = os.path.join(save_dir, safe_name)
                    torch.save(data_load, save_path)
                    print(f"   💾 Сохранено локально: {save_path}")

                if isinstance(data_load, dict):
                    if "memory" in data_load:
                        all_transitions.extend(data_load["memory"])
                    else:
                        all_transitions.append(data_load)
                elif isinstance(data_load, list):
                    all_transitions.extend(data_load)
                else:
                    print(f"   ⚠️ Формат {filename} не распознан ({type(data_load)}), пропуск.")
                    continue

                print(f"   ⬇️ {filename} загружен ({len(all_transitions)} переходов накоплено).")

            except Exception as e:
                print(f"   ❌ Ошибка при чтении {filename}: {e}")
    # tolerance =0.0608023 
    # prev_tol= 0.060776
    if tolerance > prev_tol:
        all_transitions = truncate_success_chains(all_transitions, current_tol=tolerance, prev_tol= prev_tol)

    # --- Сдвиг наград для успешных переходов ---
    all_transitions = shift_done_rewards(all_transitions, shift_value=50)

    print(f"\n🚀 Всего собрано {len(all_transitions)} переходов из {len(experiments_sorted_duration)} экспериментов.")
    if not all_transitions:
        print("⚠️ Не найдено переходов для загрузки — возвращаем пустой буфер.")
        return PrioritizedReplayBuffer(capacity=1)

    # === Заполняем буфер ===
    replay_buffer = load_transitions_to_replay_buffer(replay_buffer, all_transitions, prev_tol=prev_tol, current_tol=tolerance)

    # print(f"\n✅ Финальный буфер содержит {len(replay_buffer)} переходов.")
    return replay_buffer


def shift_done_rewards(transitions, shift_value=50):
    """
    Увеличивает model_reward на shift_value для всех переходов, где done == 1.
    Возвращает изменённый список transitions.
    """

    for tr in transitions:
        if int(tr.get("done", 0)) == 1:
            # Убедиться, что reward_model существует
            if "reward_model" in tr:
                try:
                    tr["reward_model"] = float(tr["reward_model"]) + shift_value
                except:
                    print("⚠️ Не удалось преобразовать reward_model в float:", tr["reward_model"])
            else:
                print("⚠️ У перехода нет поля reward_model", tr)

    return transitions



def truncate_success_chains(transitions, current_tol=0.0608023, prev_tol= 0.060776):
    """
    transitions: общий список переходов, отсортированный последовательно.
    Каждый эпизод заканчивается done = -1.
    Нужно: если reward < threshold → done = 1 + удалить все последующие в эпизоде.
    
    Возвращает новый список переходов.
    """


    cleaned = []
    episode = []
    flag_is_tail = False  # флаг, что мы в "хвосте" после успешного перехода

    for tr in transitions:
        if not flag_is_tail:
            episode.append(tr)
        else:
            print("⚠️ Пропускаем переход в хвосте после успешного завершения.")
            print(tr['reward'], tr['done'])

        reward = float(tr["reward"])
        done = int(tr["done"])

        if done == 1:
            cleaned.extend(episode)
            episode = []  # конец эпизода
            flag_is_tail = False
            continue

        # --- Успешный переход ---
        if prev_tol < abs(reward) <= current_tol:
            print("\n=== ⚙️ Data before modification ===")
            print({
                'reward': tr.get('reward'),
                'reward_model': tr.get('reward_model'),
                'done': tr.get('done'),
                'opt_model_i': tr.get('opt_model_i')
            })


            tr["done"] = 1
            tr["reward_model"] += 10
            cleaned.extend(episode)
            episode = []  # начать новый эпизод
            print("=== ✅ Data after modification ===")
            print({
                'reward': tr.get('reward'),
                'reward_model': tr.get('reward_model'),
                'done': tr.get('done'),
                'opt_model_i': tr.get('opt_model_i')
            })
            print("=" * 50)
            flag_is_tail = True
            continue

        # --- Конец эпизода ---
        if done == -1:
            if not flag_is_tail:
                cleaned.extend(episode)
                episode = []
            else:
                episode = []
            flag_is_tail = False

    # Если последний эпизод не завершился done=-1 — отбрасываем "хвост"
    # (позиционные ошибки уровня tolerance точно не должны жить вечно)
    
    return cleaned



# === Точка входа ===
# if __name__ == "__main__":
#     buffer = collect_all_comet_transitions(PrioritizedReplayBuffer(capacity=100000), 15)
#     torch.save(buffer.memory, "merged_replay_buffer.pt")
#     print("💾 Буфер сохранён в merged_replay_buffer.pt")
    # exp = api.get_experiment(workspace=WORKSPACE, project_name=PROJECT_NAME, experiment='751c7ca595dd4dafb22a0cfe61c26b6f')
    # meta = exp.get_metadata()
    # exp_id = meta.get("experimentKey")
    # exp_name = meta.get("experimentName")

    # assets = exp.get_asset_list()
    # pt_assets = [a for a in assets if a["fileName"].endswith(".pt") and "entry_step" in a["fileName"]]

    # for asset in pt_assets:
    #     filename = asset["fileName"]
    #     print(asset)