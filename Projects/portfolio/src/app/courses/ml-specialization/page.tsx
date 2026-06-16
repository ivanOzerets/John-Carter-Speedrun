import CoursePage from "@/app/components/course-page";
import type { Course } from "@/app/components/course-page";

const COURSES: Course[] = [
    {
        title: "Course 1: Supervised Machine Learning — Regression and Classification",
        weeks: [
            { title: "Week 1: Introduction to Machine Learning", file: "c1-week1.md" },
            { title: "Week 2: Regression with Multiple Input Variables", file: "c1-week2.md" },
            { title: "Week 3: Classification", file: "c1-week3.md" },
        ],
    },
    {
        title: "Course 2: Advanced Learning Algorithms",
        weeks: [
            {
                title: "Week 1: Neural Networks",
                file: "c2-week1.md",
                labs: [
                    { title: "Optional Lab — Neurons and Layers", file: "c2-week1-lab-neurons-and-layers.md" },
                    { title: "Optional Lab — Simple Neural Network", file: "c2-week1-lab-simple-neural-network.md" },
                    { title: "Optional Lab — Simple NN", file: "c2-week1-lab-simple-nn.md" },
                    { title: "Practice Lab — NN for Handwritten Digit Recognition", file: "c2-week1-practice-lab.md" },
                ],
            },
            { title: "Week 2: NN Training", file: null },
            { title: "Week 3: Advice for Applying Machine Learning", file: null },
            { title: "Week 4: Decision Trees", file: null },
        ],
    },
    {
        title: "Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning",
        weeks: [
            { title: "Week 1: Unsupervised Learning", file: null },
            { title: "Week 2: Recommender Systems", file: null },
            { title: "Week 3: Reinforcement Learning", file: null },
        ],
    },
];

export default function MLSpecializationPage() {
    return (
        <CoursePage
            title="Machine Learning Specialization"
            provider="Coursera · DeepLearning.AI"
            courses={COURSES}
            basePath="/notes/ml-specialization/"
        />
    );
}
