import os


def init_wandb(wandb, config):
    os.environ["WANDB_SILENT"] = "true"
    wandb.init(magic=False, project=config["tracking_project"], config=config)


class WandbLogger:
    def __init__(self, wandb, run_name=None):
        self.run_name = run_name
        self.wandb = wandb

        self.wandb.run.name = f"{self.run_name}"
        self.wandb.define_metric("metrics/_epoch")  # Establish a custom x-axis identifier used here and in log call

        # First param is a regex or specific Group/Chart that you will control
        # Second param is the custom x-axis you will use on that Chart
        self.wandb.define_metric("metrics/*", step_metric="metrics/_epoch")

    def log(self, msg):
        """
        Pass through for logging

        Parameters
        ----------
        msg  :  dict
            item being logged
        """
        self.wandb.log(msg)

    def eplog(self, epoch, title, msg):
        """
        Log against epoch x-axis instead of the default step

        Parameters
        ----------
        epoch  :  int
            epoch number
        title  :  str
            title
        msg  :  float
            item being logged
        """
        self.wandb.log({
            "metrics/_epoch": epoch,  # custom x-axis defined in custom_metric in global main
            f"metrics/{title}": msg
        })
