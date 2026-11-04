from common import ROOT, sha

def algorithm_class(pc):
    torch = pc.torch
    class TracedGNN(pc.IdentityGNN):
        """Native PyG 2.5.2 training order plus atomic optimizer checkpoints."""
        def __init__(self, ids, seed, epochs, checkpoint=None, callback=None, interval=250):
            super().__init__(ids, seed)
            self.epochs = epochs
            self.checkpoint = checkpoint
            self.callback = callback
            self.interval = interval
            self.snapshots = []

        def _train(self, model, x, edge_index, *, target, index=None, **kwargs):
            self._initialize_masks(x, edge_index)
            pc.set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)
            start = 0
            if self.checkpoint is not None and self.checkpoint.exists():
                saved = torch.load(self.checkpoint, map_location=x.device, weights_only=False)
                assert saved['seed'] == self.seed and saved['protocol_sha256'] == sha(ROOT/'protocol.json')
                assert pc.np.array_equal(saved['ids'], self.ids)
                self.edge_mask.data.copy_(saved['raw'])
                self.hard_edge_mask = saved['hard']
                optimizer.load_state_dict(saved['optimizer'])
                self.trace = saved['trace']
                self.snapshots = saved['snapshots']
                start = saved['step']
                torch.set_rng_state(saved['torch_rng'].cpu())
                if x.is_cuda:
                    torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
            for i in range(start, self.epochs):
                optimizer.zero_grad()
                y_hat, y = model(x, edge_index, **kwargs), target
                if index is not None:
                    y_hat, y = y_hat[index], y[index]
                loss = self._loss(y_hat, y)
                assert torch.isfinite(loss), 'Nonfinite native explainer objective'
                loss.backward()
                assert self.edge_mask.grad is not None and torch.isfinite(self.edge_mask.grad).all()
                optimizer.step()
                if i == 0:
                    self.hard_edge_mask = self.edge_mask.grad != 0.0
                step = i + 1
                if step % self.interval == 0 or step == self.epochs:
                    raw = self.edge_mask.detach().cpu().numpy().copy()
                    hard = self.hard_edge_mask.detach().cpu().numpy().copy()
                    ranked = raw.copy(); ranked[~hard] = -pc.np.inf
                    order = pc.np.lexsort((self.ids, -ranked))
                    snap = dict(step=step, top5=self.ids[order[:5]].tolist(), top25=self.ids[order[:25]].tolist(),
                                total_loss=self.trace[-1], loss_change_last100_fraction=(self.trace[-100]-self.trace[-1])/max(abs(self.trace[-100]),1e-12) if step>=100 else None)
                    self.snapshots.append(snap)
                    if self.callback:
                        self.callback(self, step, raw, hard, snap)
                    if self.checkpoint is not None:
                        saved = dict(step=step, seed=self.seed, ids=self.ids, raw=self.edge_mask.detach(), hard=self.hard_edge_mask,
                                     optimizer=optimizer.state_dict(), trace=self.trace, snapshots=self.snapshots,
                                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if x.is_cuda else [],
                                     protocol_sha256=sha(ROOT/'protocol.json'))
                        tmp = self.checkpoint.with_suffix('.tmp')
                        torch.save(saved, tmp); tmp.replace(self.checkpoint)
            self.saved_raw = self.edge_mask.detach().cpu().numpy().copy()
            self.saved_hard = self.hard_edge_mask.detach().cpu().numpy().copy()
    return TracedGNN

