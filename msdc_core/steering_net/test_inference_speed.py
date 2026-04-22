from time import perf_counter
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("Using GPU for inference.")
else:
    print("Using CPU for inference.")
from msdc_core.steering_net.steering_net import SteeringNet


def test_inference_speed(model_path: str, num_iterations: int = 100):
    print("Loading model from path:", model_path)
    model = SteeringNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Testing inference speed...")
    inference_times = []
    for i in range(num_iterations):
        input_tensor = torch.randn(1, 3, 240, 320).to(device)  # Simulate a batch of 1 image
        with torch.no_grad():
            start = perf_counter()
            output = model(input_tensor)
            end = perf_counter()
            inference_times.append(end - start)

        if i % 10 == 0:
            print(f"Iteration {i+1}/{num_iterations}")
    print("Tests complete.")

    # Display average inference time
    avg_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time over {num_iterations} iterations: {avg_time:.4f} seconds")
    print(f"Average FPS: {1/avg_time:.2f}")


if __name__ == "__main__":
    model_path = "/home/jetson/models/trial_03_augmented_02/train_01/steering_net_epoch_5.pth"  # Update this path to your model
    test_inference_speed(model_path, 100)
