//class MyThread extends Thread {
//    @Override // It is good practice to keep this uncommented
//    public void run() {
//        for (int i = 1; i <= 5; i++) {
//            System.out.println("Custom Thread (t1) is running: Step " + i);
//            try {
//                // Pauses the thread for 1000 milliseconds (1 second)
//                Thread.sleep(1000);
//            } catch (InterruptedException e) {
//                System.out.println("Thread was interrupted.");
//            }
//        }
//        System.out.println("--- Custom Thread finished. ---");
//    }
//}
//
//public class MThread {
//    public static void main(String[] args) {
//        System.out.println("--- Main Thread started. ---");
//
//        MyThread t1 = new MyThread();
//        t1.start(); // Starts the custom thread in the background
//
//        // Let's also make the Main thread do some work simultaneously
//        for (int i = 1; i <= 3; i++) {
//            System.out.println("Main Thread is running: Step " + i);
//            try {
//                Thread.sleep(1000);
//            } catch (InterruptedException e) {
//                System.out.println("Main Thread was interrupted.");
//            }
//        }
//        System.out.println("--- Main Thread finished. ---");
//    }
//}

import java.util.*;
class MyThread extends Thread{
    public void run(){
        System.out.println("Thread is running ");
    }
}
public class MThread {
    public static void main(String[] args){
        MyThread t = new MyThread();
        t.start();
    }
}