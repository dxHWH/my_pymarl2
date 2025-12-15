import torch
import torch.optim as optim
import torch.nn.functional as F # <-- 确保 F 已导入
import os
import logging
from tqdm import tqdm

# 确保这里的导入路径与您的文件结构一致
# 假设您的 vae.py 在一个名为 world 的文件夹下
from world.vae import vae_loss_function 

logger = logging.getLogger(__name__)

class VAETrainer:
    def __init__(self, model, optimizer, train_dataloader, test_dataloader, device, config):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        
        self.total_train_steps = config.get('total_train_steps', 50000)
        self.eval_every_n_steps = config.get('eval_every_n_steps', 100)
        self.eval_n_batches = config.get('eval_n_batches', 100) # (新)
        
        self.save_dir = config.get('save_dir', './vae_models')
        self.model_name = config.get('model_name', 'vae_model')
        
        self.best_test_loss = float('inf')
        self.train_iter = iter(self.train_dataloader)
        
        # --- (已修复) ---
        # (新) 创建可重复使用的测试迭代器
        if self.test_dataloader:
            self.test_iter = iter(self.test_dataloader)
        # --- (修复结束) ---
        
        logger.info("VAETrainer initialized.")


    def _train_step(self):
        """
        (此函数已正确) 执行一个训练步骤，并返回三个“每样本”损失
        """
        self.model.train() 
        try:
            encoder_input, state_t_plus_1 = next(self.train_iter)
        except StopIteration:
            logger.info("Training dataloader exhausted. Reshuffling and restarting...")
            self.train_iter = iter(self.train_dataloader)
            encoder_input, state_t_plus_1 = next(self.train_iter)
            
        current_batch_size = encoder_input.shape[0]

        encoder_input = encoder_input.to(self.device)
        state_t_plus_1 = state_t_plus_1.to(self.device)
        
        recon_state, mu, log_var = self.model(encoder_input)
        
        # (已更新) 获取三个损失分量
        loss, rce, kld = vae_loss_function(recon_state, state_t_plus_1, mu, log_var)
        
        self.optimizer.zero_grad()
        loss.backward() # 只在总损失上反向传播
        self.optimizer.step()
        
        # (已更新) 返回三个“每样本”平均损失 (修复了日志 Bug)
        return (
            loss.item() / current_batch_size, 
            rce.item() / current_batch_size, 
            kld.item() / current_batch_size
        )

    def _evaluate(self):
        """
        (已修复) 在测试集上评估固定数量的批次
        """
        self.model.eval() 
        total_vae_loss = 0
        total_rce_loss = 0
        total_kld_loss = 0
        total_samples_processed = 0 # (新)
        
        all_preds = []
        all_truths = []
        
        # eval_pbar 迭代的是 range(self.eval_n_batches)
        eval_pbar = tqdm(range(self.eval_n_batches), desc="Evaluating", leave=False, colour="green")
        
        with torch.no_grad():
            # --- (这是已修复的循环) ---
            # i 只是一个整数, 比如 0, 1, 2...
            for i in eval_pbar:
                try:
                    # --- (这是正确的数据解包) ---
                    # 我们在循环内部从迭代器中获取数据
                    encoder_input, state_t_plus_1 = next(self.test_iter)
                except StopIteration:
                    # 如果测试集耗尽，重置它
                    self.test_iter = iter(self.test_dataloader)
                    encoder_input, state_t_plus_1 = next(self.test_iter)

                current_batch_size = encoder_input.shape[0]
                total_samples_processed += current_batch_size # (新)

                encoder_input = encoder_input.to(self.device)
                state_t_plus_1 = state_t_plus_1.to(self.device)
                
                recon_state, mu, log_var = self.model(encoder_input)
                
                # vae_loss_function 返回的是聚合值 (sum)
                vae_loss, rce, kld = vae_loss_function(recon_state, state_t_plus_1, mu, log_var)
                
                total_vae_loss += vae_loss.item()
                total_rce_loss += rce.item()
                total_kld_loss += kld.item()
                
                all_preds.append(recon_state.cpu())
                all_truths.append(state_t_plus_1.cpu())

        # --- (已修复) 计算聚合指标 ---
        
        # (已修复) 除以实际处理的样本数
        avg_vae_loss = total_vae_loss / (total_samples_processed + 1e-8)
        avg_rce_loss = total_rce_loss / (total_samples_processed + 1e-8)
        avg_kld_loss = total_kld_loss / (total_samples_processed + 1e-8)
        
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_truths_tensor = torch.cat(all_truths, dim=0)
        
        avg_mae = F.l1_loss(all_preds_tensor, all_truths_tensor).item()
        
        ss_residual = ((all_truths_tensor - all_preds_tensor) ** 2).sum()
        ss_total = ((all_truths_tensor - all_truths_tensor.mean(dim=0)) ** 2).sum()
        r2_score = 1.0 - (ss_residual / (ss_total + 1e-8))
        
        metrics = {
            "vae_loss": avg_vae_loss,
            "rce_loss": avg_rce_loss,
            "kld_loss": avg_kld_loss,
            "mae": avg_mae,
            "r2_score": r2_score.item()
        }
        return metrics


    def _save_checkpoint(self, test_loss):
        # (此函数已正确)
        model_path = os.path.join(self.save_dir, f"vae_world_model_{self.model_name}_best.pt")
        torch.save(self.model.state_dict(), model_path)
        encoder_path = os.path.join(self.save_dir, f"vae_encoder_{self.model_name}_best.pt")
        encoder_state_dict = {
            k: v for k, v in self.model.state_dict().items() 
            if k.startswith('encoder_net') or k.startswith('fc_mu') or k.startswith('fc_log_var')
        }
        torch.save(encoder_state_dict, encoder_path)
        logger.info(f"  -> New best model saved to {model_path} (Test VAE Loss: {test_loss:.4f})")

    def run_training(self):
        """
        (此函数已正确) 执行完整的训练流程
        """
        logger.info("Starting VAE training...")
        logger.info(f"Total steps: {self.total_train_steps} | Evaluate every: {self.eval_every_n_steps} steps.")
        
        self.best_test_loss = float('inf')
        total_steps = 0
        num_eval_runs = (self.total_train_steps + self.eval_every_n_steps - 1) // self.eval_every_n_steps
        
        for run_idx in range(num_eval_runs):
            
            steps_in_this_run = min(self.eval_every_n_steps, self.total_train_steps - total_steps)
            pbar = tqdm(range(steps_in_this_run), 
                        desc=f"Run {run_idx+1}/{num_eval_runs} (Steps {total_steps+1}-{total_steps+steps_in_this_run})", 
                        leave=True, 
                        colour="cyan")
            
            sum_train_vae = 0.0
            sum_train_rce = 0.0
            sum_train_kld = 0.0

            for step in pbar:
                step_vae, step_rce, step_kld = self._train_step() 
                
                sum_train_vae += step_vae
                sum_train_rce += step_rce
                sum_train_kld += step_kld
                
                pbar.set_postfix({"batch_loss": f"{step_vae:.4f}"})

            total_steps += steps_in_this_run
            
            avg_train_vae = sum_train_vae / steps_in_this_run
            avg_train_rce = sum_train_rce / steps_in_this_run
            avg_train_kld = sum_train_kld / steps_in_this_run

            if self.test_dataloader:
                test_metrics = self._evaluate()
                
                log_msg_train = f"[Step {total_steps:6d}] Train VAE: {avg_train_vae:7.4f} " \
                                f"(RCE: {avg_train_rce:7.4f}, KLD: {avg_train_kld:7.4f})"
                logger.info(log_msg_train)
                
                log_msg_test =  f"[Step {total_steps:6d}] Test  VAE: {test_metrics['vae_loss']:7.4f} " \
                                f"(RCE: {test_metrics['rce_loss']:7.4f}, KLD: {test_metrics['kld_loss']:7.4f}) | " \
                                f"MAE: {test_metrics['mae']:6.4f} | R²: {test_metrics['r2_score']:6.4f}"
                logger.info(log_msg_test)
                
                current_test_loss = test_metrics['vae_loss']
            else:
                current_test_loss = avg_train_vae # Fallback
                logger.info(f"[Step {total_steps:6d}] Avg Train Loss: {avg_train_vae:.4f} | (No test data)")

            if current_test_loss < self.best_test_loss:
                self.best_test_loss = current_test_loss
                self._save_checkpoint(self.best_test_loss)

        logger.info("Training complete.")