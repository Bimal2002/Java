import java.util.*;

class Node {
    int data;
    Node next;

    Node(int data) {
        this.data = data;
        this.next = null;
    }
}

public class LinkedListdata {
    // Insert at Beginning
    Node insertStart(Node head, int data) {
        Node newNode = new Node(data);
        newNode.next = head;
        return newNode;
    }

    // Insert at End
    Node insertAtEnd(Node head, int data) {
        Node newNode = new Node(data);
        if (head == null) {
            return newNode;
        }
        Node temp = head;
        while (temp.next != null) {
            temp = temp.next;
        }
        temp.next = newNode;
        return head;
    }

    // Delete Node by Key
    Node delete(Node head, int key) {
        if (head == null) return null;

        // If head node itself holds the key to be deleted
        if (head.data == key) {
            return head.next;
        }

        Node temp = head;
        while (temp.next != null && temp.next.data != key) {
            temp = temp.next;
        }

        // If key was found, unlink the node from linked list
        if (temp.next != null) {
            temp.next = temp.next.next;
        }
        return head;
    }

    // Traversal
    void printList(Node head) {
        Node temp = head;
        while (temp != null) {
            System.out.print(temp.data + " -> ");
            temp = temp.next;
        }
        System.out.println("null");
    }

    public static void main(String[] args) {
        // Corrected: Using the actual class name "LinkedListdata"
        LinkedListdata list = new LinkedListdata();
        Node head = null;

        // 1. Insert data
        head = list.insertStart(head, 10);
        head = list.insertStart(head, 20); // List: 20 -> 10
        head = list.insertAtEnd(head, 30);  // List: 20 -> 10 -> 30
        head = list.insertAtEnd(head, 40);  // List: 20 -> 10 -> 30 -> 40

        System.out.println("Original List:");
        list.printList(head);

        // 2. Delete a node
        System.out.println("\nDeleting 10...");
        head = list.delete(head, 10);

        // 3. Print final list
        System.out.println("Final List:");
        list.printList(head);
    }
}
