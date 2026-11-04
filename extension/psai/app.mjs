import { PSAIController } from "./controller.mjs";
import { PSAIView } from "./view.mjs";

const app = new PSAIController(new PSAIView(document), {
  storage: globalThis.chrome?.storage?.local,
  openTab: (url) => chrome.tabs.create({ url }),
});

app.initialize();
