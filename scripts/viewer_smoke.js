"use strict";

const fs = require("fs");
const vm = require("vm");

const source = fs.readFileSync(process.argv[2] || "main.js", "utf8");
vm.runInThisContext(source, { filename: "main.js" });

const empty = new Field(0).disp();
const terminalPreview = preview(Math.pow(2, 40) - 1, [[4353.5]]);
if (terminalPreview !== empty) {
    throw new Error("zero-step terminal preview did not clear the completed PC field");
}

const terminalHtml = { innerHTML: "" };
const survivalHtml = { textContent: "", style: {} };
global.document = {
    getElementById(id) {
        if (id === "results") {
            return terminalHtml;
        }
        if (id === "survival") {
            return survivalHtml;
        }
        throw new Error(`unexpected element id: ${id}`);
    }
};
global.objective = "expected_pc";
global.survival_success = 839;
global.survival_total = 840;
disp_survival();
if (survivalHtml.textContent !== "Baseline survival: 839/840 (99.881%)") {
    throw new Error("survival summary was not displayed with three decimal places");
}
disp_options(Math.pow(2, 40) - 1, [[4353.5]]);
if (!terminalHtml.innerHTML.includes("4353.500")) {
    throw new Error("zero-step terminal view did not display its V* score");
}
