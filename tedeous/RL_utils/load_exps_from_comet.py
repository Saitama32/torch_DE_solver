from comet_ml import API
import torch
import io
from datetime import datetime
from tedeous.rl_algorithms import PrioritizedReplayBuffer
from tedeous.RL_utils.load_transitions_into_buffer_pickle import load_transitions_to_replay_buffer


# === Настройки ===
WORKSPACE = "saitama32"
PROJECT_NAME = "rlpinn"
# MAX_EXPERIMENTS = 15  # можно изменить при необходимости

api = API(api_key="aP71fQTYPNqfsYWvudPPmoBl5")  # или просто API()


# === Вспомогательные функции ===
def get_metadata_field(exp, field, default=None):
    try:
        meta = exp.get_metadata()
        return meta.get(field, default)
    except Exception:
        return default


def get_end_time(exp):
    end_ms = get_metadata_field(exp, "endTimeMillis")
    if end_ms:
        return datetime.fromtimestamp(end_ms / 1000)
    return datetime.min


def is_crashed(exp):
    return get_metadata_field(exp, "hasCrashed", False) is True


# === Основная функция ===
def collect_all_comet_transitions(replay_buffer=None, max_exps_last=10) -> PrioritizedReplayBuffer:
    """Собирает все переходы из не-crashed экспериментов проекта и возвращает заполненный PrioritizedReplayBuffer."""
    print("🔍 Получаем эксперименты из Comet...")
    experiments = list(api.get_experiments(workspace=WORKSPACE, project_name=PROJECT_NAME))
    experiments_sorted = sorted(experiments, key=get_end_time, reverse=True)
    # experiments_sorted = [api.get_experiment(workspace=WORKSPACE, project_name=PROJECT_NAME, experiment='751c7ca595dd4dafb22a0cfe61c26b6f')]
    valid_experiments = [exp for exp in experiments_sorted if not is_crashed(exp)]
    valid_experiments = valid_experiments[:max_exps_last]

    print(f"✅ Найдено {len(valid_experiments)} активных экспериментов для загрузки буферов.\n")

    all_transitions = []  # сюда соберём всё

    for i, exp in enumerate(valid_experiments, 1):
        meta = exp.get_metadata()
        exp_id = meta.get("experimentKey")
        exp_name = meta.get("experimentName")
        print(f"[{i:2d}] {exp_name} ({exp_id})")

        assets = exp.get_asset_list()
        pt_assets = [a for a in assets if a["fileName"].endswith(".pt") and "entry_step" in a["fileName"]]

        if not pt_assets:
            print("   ⚠️ Нет файлов entry_step_*.pt — пропускаем.")
            continue

        for asset in pt_assets:
            filename = asset["fileName"]

            try:
                file_bytes = exp.get_asset(asset["assetId"], return_type="binary")
                buffer_stream = io.BytesIO(file_bytes)
                data_load = torch.load(buffer_stream, map_location="cpu")

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

    print(f"\n🚀 Всего собрано {len(all_transitions)} переходов из {len(valid_experiments)} экспериментов.")
    if not all_transitions:
        print("⚠️ Не найдено переходов для загрузки — возвращаем пустой буфер.")
        return PrioritizedReplayBuffer(capacity=1)

    # === Заполняем буфер ===
    replay_buffer = load_transitions_to_replay_buffer(replay_buffer, all_transitions)

    # print(f"\n✅ Финальный буфер содержит {len(replay_buffer)} переходов.")
    return replay_buffer


# === Точка входа ===
if __name__ == "__main__":
    buffer = collect_all_comet_transitions(PrioritizedReplayBuffer(capacity=100000))
    # torch.save(buffer.memory, "merged_replay_buffer.pt")
    # print("💾 Буфер сохранён в merged_replay_buffer.pt")
