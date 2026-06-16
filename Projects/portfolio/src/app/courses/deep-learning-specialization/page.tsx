import CoursePage from "@/app/components/course-page";
import type { Course } from "@/app/components/course-page";

const COURSES: Course[] = [
    {
        title: "Course 1: Neural Networks and Deep Learning",
        weeks: [
            { title: "Week 1: Introduction to Deep Learning", file: null },
            { title: "Week 2: Neural Networks Basics", file: null },
            { title: "Week 3: Shallow neural Networks", file: null },
            { title: "Week 4: Deep Neural Networks", file: null },
        ],
    },
    {
        title: "Course 2: Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization",
        weeks: [
            { title: "Week 1: Practical Aspects of Deep Learning", file: null },
            { title: "Week 2: Optimization Algorithms", file: null },
            { title: "Week 3: Hyperparameter Tuning, Batch and Programming Frameworks", file: null },
        ],
    },
    {
        title: "Course 3: Structuring Machine Learning Projects",
        weeks: [
            { title: "Week 1: ML Strategy", file: null },
            { title: "Week 2: ML Strategy", file: null },
        ],
    },
    {
        title: "Course 4: Convolutional Neural Networks",
        weeks: [
            { title: "Week 1: Foundations of Convolutional Neural Networks", file: null },
            { title: "Week 2: Deep Convolutional Neural Networks: Case Studies", file: null },
            { title: "Week 3: Object Detection", file: null },
            { title: "Week 4: Special Applications: Face Recognition and Neural Style Transfer", file: null },
        ],
    },
    {
        title: "Course 5: Sequence Models",
        weeks: [
            { title: "Week 1: Recurrent Neural Networks", file: null },
            { title: "Week 2: Natural Language Processing and Word Embeddings", file: null },
            { title: "Week 3: Sequence Models and Attention Mechanism", file: null },
            { title: "Week 4: Transformer Network", file: null },
        ],
    },
];

export default function DeepLearningPage() {
    return (
        <CoursePage
            title="Deep Learning Specialization"
            provider="Coursera · DeepLearning.AI"
            courses={COURSES}
            basePath="/notes/deep-learning-specialization/"
        />
    );
}