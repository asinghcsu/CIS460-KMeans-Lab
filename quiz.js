// Student knowledge check. Results stay on the page and are not saved or sent anywhere.
(() => {
  "use strict";

  const QUESTIONS = [
    {
      text: "What does K represent in K-Means?",
      options: ["The number of features", "The number of clusters", "The number of iterations"],
      answer: 1,
      why: "K is the number of clusters, which the analyst sets before the algorithm runs.",
    },
    {
      text: "After assigning points to clusters, what happens next?",
      options: ["The centers move to the group means", "The most distant points are discarded", "The value of K automatically increases"],
      answer: 0,
      why: "The update step moves each center to the mean of its points. Assignment then repeats until no point changes cluster.",
    },
    {
      text: "Why can two K-Means runs give different results?",
      options: ["The distances are always random", "Different starting centers can lead to different local optima", "The algorithm changes K automatically"],
      answer: 1,
      why: "The result depends on the start. In the lab, the exam scores end at inertia 186.08 from a well-spread start and 1005.43 from a poor one.",
    },
    {
      text: "What is the main advantage of DBSCAN on two moons?",
      options: ["It follows density-connected curved shapes", "It always finds more clusters", "It does not use distances"],
      answer: 0,
      why: "DBSCAN grows clusters through connected dense regions, so it can follow a curve. It still uses distances, through eps.",
    },
  ];

  const $ = (id) => document.getElementById(id);
  const form = $("quizForm");

  form.querySelector(".quiz-questions").innerHTML = QUESTIONS.map((q, i) => `
    <fieldset class="quiz-q" data-q="${i}">
      <legend>${i + 1}. ${q.text}</legend>
      ${q.options.map((option, j) => `<label class="quiz-option"><input type="radio" name="q${i}" value="${j}"> <span>${option}</span></label>`).join("")}
      <p class="quiz-feedback" hidden></p>
    </fieldset>`).join("");

  const chosen = (i) => {
    const input = form.querySelector(`input[name="q${i}"]:checked`);
    return input ? Number(input.value) : null;
  };
  const answered = () => QUESTIONS.filter((_, i) => chosen(i) !== null).length;

  function clearResults() {
    form.querySelectorAll(".quiz-feedback").forEach((p) => { p.hidden = true; p.textContent = ""; });
    form.querySelectorAll(".quiz-q").forEach((f) => f.classList.remove("is-right", "is-wrong"));
    $("quizScore").textContent = "";
  }

  function updateProgress() {
    const n = answered();
    $("quizProgress").textContent = `${n} of ${QUESTIONS.length} answered`;
    $("quizCheck").disabled = n < QUESTIONS.length;
  }

  form.addEventListener("change", () => { clearResults(); updateProgress(); });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    let score = 0;
    QUESTIONS.forEach((q, i) => {
      const right = chosen(i) === q.answer;
      if (right) score += 1;
      const box = form.querySelector(`.quiz-q[data-q="${i}"]`);
      box.classList.toggle("is-right", right);
      box.classList.toggle("is-wrong", !right);
      const feedback = box.querySelector(".quiz-feedback");
      feedback.textContent = `${right ? "Correct." : `Incorrect. The answer is: ${q.options[q.answer]}.`} ${q.why}`;
      feedback.hidden = false;
    });
    $("quizScore").textContent = `Score: ${score} of ${QUESTIONS.length}`;
  });

  form.addEventListener("reset", () => {
    // The browser clears the radio buttons after this event finishes.
    setTimeout(() => { clearResults(); updateProgress(); }, 0);
  });

  updateProgress();
})();
